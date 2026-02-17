import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Literal, Optional

import httpx
import redis.asyncio as redis
import jwt
from redis.exceptions import RedisError
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .storage import BetRecord, IdempotencyRecord, SessionLocal, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
audit_logger = logging.getLogger("audit")

MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
IDEMPOTENCY_TTL_HOURS = int(os.getenv("IDEMPOTENCY_TTL_HOURS", "24"))
EXTERNAL_REQUEST_TIMEOUT_SECONDS = float(os.getenv("EXTERNAL_REQUEST_TIMEOUT_SECONDS", "5"))
IDEMPOTENCY_WAIT_SECONDS = float(os.getenv("IDEMPOTENCY_WAIT_SECONDS", "1.0"))
CIRCUIT_BREAKER_FAILURES = int(os.getenv("CIRCUIT_BREAKER_FAILURES", "5"))
CIRCUIT_BREAKER_OPEN_SECONDS = int(os.getenv("CIRCUIT_BREAKER_OPEN_SECONDS", "45"))
ENABLE_HSTS = os.getenv("ENABLE_HSTS", "false").lower() == "true"
INTERNAL_GATEWAY_SECRET = os.getenv("INTERNAL_GATEWAY_SECRET")
AUTH_SECRET = os.getenv("AUTH_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_JWT_AUDIENCE = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")
SUPABASE_JWT_ISSUER = os.getenv("SUPABASE_JWT_ISSUER")
SUPABASE_JWKS_CACHE_TTL_SECONDS = int(os.getenv("SUPABASE_JWKS_CACHE_TTL_SECONDS", "300"))
SUPABASE_JWKS_REQUEST_TIMEOUT_SECONDS = float(os.getenv("SUPABASE_JWKS_REQUEST_TIMEOUT_SECONDS", "3"))
ENVIRONMENT = (os.getenv("ENVIRONMENT") or "development").lower()
REDIS_URL = os.getenv("REDIS_URL")
ENABLE_RATE_LIMITING = bool(REDIS_URL)
REDIS_CONNECT_TIMEOUT_SECONDS = float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS", "1.5"))
REDIS_SOCKET_TIMEOUT_SECONDS = float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "1.5"))
REDIS_HEALTH_CHECK_INTERVAL_SECONDS = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL_SECONDS", "30"))
RATE_LIMITER_RETRY_SECONDS = float(os.getenv("RATE_LIMITER_RETRY_SECONDS", "15"))
RATE_LIMITER_WARNING_INTERVAL_SECONDS = float(os.getenv("RATE_LIMITER_WARNING_INTERVAL_SECONDS", "30"))
LOCAL_RATE_LIMITER_MAX_BUCKETS = int(os.getenv("LOCAL_RATE_LIMITER_MAX_BUCKETS", "50000"))
RATE_LIMITER_REQUEST_TIMEOUT_SECONDS = float(os.getenv("RATE_LIMITER_REQUEST_TIMEOUT_SECONDS", "2.5"))

if not SUPABASE_JWT_ISSUER and SUPABASE_URL:
    SUPABASE_JWT_ISSUER = f"{SUPABASE_URL.rstrip('/')}/auth/v1"
SUPABASE_JWKS_URL = (
    f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json" if SUPABASE_URL else None
)

if not SUPABASE_URL and not AUTH_SECRET and ENVIRONMENT != "production":
    AUTH_SECRET = "dev-auth-secret-change-me"
    logger.warning(
        "SUPABASE_URL and AUTH_SECRET are missing; using development AUTH_SECRET fallback. "
        "Do not use this in production."
    )

def parse_limit(value: str, default_times: int, default_seconds: int) -> dict:
    raw = os.getenv(value)
    if not raw:
        return {"times": default_times, "seconds": default_seconds}
    try:
        times_str, seconds_str = raw.split("/", 1)
        return {"times": int(times_str), "seconds": int(seconds_str)}
    except ValueError:
        return {"times": default_times, "seconds": default_seconds}


LIMITS_DEFAULT = parse_limit("RATE_LIMIT_DEFAULT", 300, 60)
LIMITS_AUTH = parse_limit("RATE_LIMIT_AUTH", 5, 60)
LIMITS_ODDS = parse_limit("RATE_LIMIT_ODDS", 30, 60)
LIMITS_BETS = parse_limit("RATE_LIMIT_BETS", 5, 60)
LIMITS_PAYOUTS = parse_limit("RATE_LIMIT_PAYOUTS", 3, 60)
LIMITS_READ = parse_limit("RATE_LIMIT_READ", 120, 60)

PROTECTED_PATHS = {"/bets", "/payouts"}


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def get_user_id(request: Request) -> Optional[str]:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return str(user_id)
    return None


def require_user_id(request: Request) -> str:
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user_id


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


async def _fetch_jwks_by_kid() -> dict[str, dict]:
    if not SUPABASE_JWKS_URL:
        raise RuntimeError("Missing SUPABASE_JWKS_URL. Set SUPABASE_URL for JWT verification.")

    async with httpx.AsyncClient(timeout=SUPABASE_JWKS_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(SUPABASE_JWKS_URL)
        response.raise_for_status()
        payload = response.json()

    keys = payload.get("keys")
    if not isinstance(keys, list):
        raise RuntimeError("Invalid JWKS payload from Supabase.")

    parsed: dict[str, dict] = {}
    for key in keys:
        if not isinstance(key, dict):
            continue
        kid = key.get("kid")
        if isinstance(kid, str) and kid:
            parsed[kid] = key

    if not parsed:
        raise RuntimeError("Supabase JWKS payload did not include signing keys.")

    return parsed


jwks_cache_by_kid: dict[str, dict] = {}
jwks_cache_expires_at = 0.0
jwks_cache_lock: Optional[asyncio.Lock] = None


async def _get_cached_jwk(kid: str) -> Optional[dict]:
    global jwks_cache_by_kid, jwks_cache_expires_at, jwks_cache_lock

    if not SUPABASE_JWKS_URL:
        return None

    now = time.monotonic()
    if kid in jwks_cache_by_kid and now < jwks_cache_expires_at:
        return jwks_cache_by_kid[kid]

    if jwks_cache_lock is None:
        jwks_cache_lock = asyncio.Lock()

    async with jwks_cache_lock:
        now = time.monotonic()
        if kid in jwks_cache_by_kid and now < jwks_cache_expires_at:
            return jwks_cache_by_kid[kid]
        try:
            jwks_cache_by_kid = await _fetch_jwks_by_kid()
            jwks_cache_expires_at = time.monotonic() + SUPABASE_JWKS_CACHE_TTL_SECONDS
        except Exception as exc:
            logger.warning("supabase_jwks_fetch_failed error=%s", str(exc))
            if kid in jwks_cache_by_kid:
                return jwks_cache_by_kid[kid]
            return None

    return jwks_cache_by_kid.get(kid)


def _verify_legacy_bearer_token(token: str) -> Optional[str]:
    if not AUTH_SECRET:
        return None
    if ":" not in token:
        return None
    user_id, signature = token.split(":", 1)
    if not user_id or not signature:
        return None
    expected = hmac.new(AUTH_SECRET.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    return user_id


async def verify_bearer_token(token: str) -> Optional[str]:
    if SUPABASE_JWKS_URL and SUPABASE_JWT_ISSUER:
        try:
            headers = jwt.get_unverified_header(token)
        except Exception:
            return None

        kid = headers.get("kid")
        if not isinstance(kid, str) or not kid:
            return None

        jwk = await _get_cached_jwk(kid)
        if not jwk:
            return None

        try:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
            if SUPABASE_JWT_AUDIENCE:
                claims = jwt.decode(
                    token,
                    key=public_key,
                    algorithms=["RS256"],
                    audience=SUPABASE_JWT_AUDIENCE,
                    issuer=SUPABASE_JWT_ISSUER,
                    options={"require": ["sub", "exp"]},
                )
            else:
                claims = jwt.decode(
                    token,
                    key=public_key,
                    algorithms=["RS256"],
                    issuer=SUPABASE_JWT_ISSUER,
                    options={"verify_aud": False, "require": ["sub", "exp"]},
                )
        except jwt.InvalidTokenError:
            return None

        subject = claims.get("sub")
        if isinstance(subject, str) and subject:
            return subject
        return None

    return _verify_legacy_bearer_token(token)


def sign_user_id(user_id: str) -> str:
    if not AUTH_SECRET:
        raise RuntimeError("Missing AUTH_SECRET. Set it to enable authenticated endpoints.")
    signature = hmac.new(AUTH_SECRET.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{user_id}:{signature}"


def hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def rate_limit_identifier(request: Request) -> str:
    user_id = get_user_id(request)
    if user_id:
        return f"user:{user_id}"
    return f"ip:{get_client_ip(request)}"


async def rate_limit_callback(request: Request, _response):
    user_id = get_user_id(request) or "-"
    ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "-")
    logger.warning(
        "rate_limited path=%s user_id=%s ip=%s ua=%s request_id=%s",
        request.url.path,
        user_id,
        ip,
        user_agent,
        get_request_id(request),
    )
    return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})


rate_limiter_ready = False
rate_limiter_next_retry_at = 0.0
rate_limiter_last_warning_at = 0.0
rate_limiter_init_lock: Optional[asyncio.Lock] = None
rate_limiter_client: Optional[redis.Redis] = None
# Emergency in-process limiter used when Redis is unavailable.
local_rate_limiter_lock: Optional[asyncio.Lock] = None
local_rate_limiter_hits: dict[str, List[float]] = {}


def _rate_limiter_warning_allowed() -> bool:
    global rate_limiter_last_warning_at
    now = time.monotonic()
    if now - rate_limiter_last_warning_at < RATE_LIMITER_WARNING_INTERVAL_SECONDS:
        return False
    rate_limiter_last_warning_at = now
    return True


def _build_local_rate_limit_key(request: Request, limits: dict) -> str:
    actor = get_user_id(request) or get_client_ip(request)
    return f"{request.url.path}|{actor}|{limits['times']}|{limits['seconds']}"


def _prune_local_rate_limit_buckets(now: float) -> None:
    stale_keys = []
    for key, timestamps in local_rate_limiter_hits.items():
        if not timestamps:
            stale_keys.append(key)
            continue
        if now - timestamps[-1] > 3600:
            stale_keys.append(key)
    for key in stale_keys:
        local_rate_limiter_hits.pop(key, None)


async def _apply_local_rate_limit(request: Request, limits: dict) -> None:
    global local_rate_limiter_lock
    if local_rate_limiter_lock is None:
        local_rate_limiter_lock = asyncio.Lock()

    now = time.monotonic()
    window_seconds = float(limits["seconds"])
    max_hits = int(limits["times"])
    key = _build_local_rate_limit_key(request, limits)

    async with local_rate_limiter_lock:
        cutoff = now - window_seconds
        timestamps = [ts for ts in local_rate_limiter_hits.get(key, []) if ts > cutoff]

        if len(timestamps) >= max_hits:
            logger.warning(
                "local_rate_limited path=%s user_id=%s ip=%s request_id=%s",
                request.url.path,
                get_user_id(request) or "-",
                get_client_ip(request),
                get_request_id(request),
            )
            raise HTTPException(status_code=429, detail="Too Many Requests")

        timestamps.append(now)
        local_rate_limiter_hits[key] = timestamps

        if len(local_rate_limiter_hits) > LOCAL_RATE_LIMITER_MAX_BUCKETS:
            _prune_local_rate_limit_buckets(now)


def _build_redis_client(redis_url: str):
    if redis_url.startswith("fakeredis://"):
        from fakeredis.aioredis import FakeRedis

        return FakeRedis(decode_responses=True)
    return redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        health_check_interval=REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
    )


async def _close_rate_limiter_client() -> None:
    global rate_limiter_client
    if rate_limiter_client is None:
        return
    try:
        await rate_limiter_client.close()
    except Exception:
        logger.exception("rate_limiter_redis_close_failed")
    finally:
        rate_limiter_client = None


async def _mark_rate_limiter_unavailable(
    reason: str,
    error: Exception,
    request: Optional[Request] = None,
) -> None:
    global rate_limiter_ready, rate_limiter_next_retry_at
    rate_limiter_ready = False
    rate_limiter_next_retry_at = time.monotonic() + RATE_LIMITER_RETRY_SECONDS
    await _close_rate_limiter_client()

    if _rate_limiter_warning_allowed():
        request_path = request.url.path if request else "-"
        logger.warning(
            "rate_limiter_unavailable reason=%s path=%s retry_in=%.1fs local_fallback=enabled error=%s",
            reason,
            request_path,
            RATE_LIMITER_RETRY_SECONDS,
            str(error),
        )


async def _initialize_rate_limiter(force: bool = False) -> bool:
    global rate_limiter_ready, rate_limiter_next_retry_at, rate_limiter_init_lock, rate_limiter_client

    if not ENABLE_RATE_LIMITING or not REDIS_URL:
        return False

    if rate_limiter_ready:
        return True

    now = time.monotonic()
    if not force and now < rate_limiter_next_retry_at:
        return False

    if rate_limiter_init_lock is None:
        rate_limiter_init_lock = asyncio.Lock()

    async with rate_limiter_init_lock:
        if rate_limiter_ready:
            return True
        now = time.monotonic()
        if not force and now < rate_limiter_next_retry_at:
            return False

        try:
            client = _build_redis_client(REDIS_URL)
            await client.ping()
            await FastAPILimiter.init(
                client,
                identifier=rate_limit_identifier,
                http_callback=rate_limit_callback,
            )
            rate_limiter_client = client
            rate_limiter_ready = True
            rate_limiter_next_retry_at = 0.0
            logger.info("rate_limiter_initialized")
            return True
        except Exception as exc:
            await _mark_rate_limiter_unavailable("init_failed", exc)
            return False


class CircuitBreaker:
    def __init__(self, failures: int, open_seconds: int) -> None:
        self.failures = failures
        self.open_seconds = open_seconds
        self.failure_count = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        if time.monotonic() < self.open_until:
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.open_until = 0.0

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failures:
            self.open_until = time.monotonic() + self.open_seconds


odds_breaker = CircuitBreaker(CIRCUIT_BREAKER_FAILURES, CIRCUIT_BREAKER_OPEN_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not REDIS_URL:
        if ENVIRONMENT == "production":
            raise RuntimeError("Missing REDIS_URL. Set it to enable shared rate limiting.")
        logger.warning("REDIS_URL missing, rate limiting disabled in development.")

    if ENVIRONMENT == "production" and not SUPABASE_URL:
        raise RuntimeError("Missing SUPABASE_URL. Set it to verify Supabase access tokens.")
    if not SUPABASE_URL and AUTH_SECRET:
        logger.warning("SUPABASE_URL missing; using legacy AUTH_SECRET bearer verification.")

    if ENABLE_RATE_LIMITING:
        initialized = await _initialize_rate_limiter(force=True)
        if not initialized and ENVIRONMENT == "production":
            logger.error(
                "rate_limiter_startup_failed: API is running fail-open and will retry Redis initialization."
            )

    init_db()
    try:
        yield
    finally:
        await _close_rate_limiter_client()

def _noop_limiter():
    return None

def limiter_dep(limits: dict):
    if not ENABLE_RATE_LIMITING:
        return Depends(_noop_limiter)

    limiter = RateLimiter(**limits)

    async def _rate_limiter_dependency(request: Request, response: Response):
        ready = await _initialize_rate_limiter()
        if not ready:
            await _apply_local_rate_limit(request, limits)
            return None

        try:
            return await asyncio.wait_for(
                limiter(request, response),
                timeout=RATE_LIMITER_REQUEST_TIMEOUT_SECONDS,
            )
        except (RedisError, asyncio.TimeoutError, TimeoutError, OSError) as exc:
            await _mark_rate_limiter_unavailable("request_failed", exc, request=request)
            await _apply_local_rate_limit(request, limits)
            return None
        except HTTPException:
            raise
        except Exception as exc:
            await _mark_rate_limiter_unavailable("request_unexpected_error", exc, request=request)
            await _apply_local_rate_limit(request, limits)
            return None

    return Depends(_rate_limiter_dependency)

app = FastAPI(
    lifespan=lifespan,
    dependencies=[limiter_dep(LIMITS_DEFAULT)],
)

PRODUCTION_WEB_ORIGINS = [
    "https://udfall.com",
    "https://www.udfall.com",
]
DEVELOPMENT_WEB_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if ENVIRONMENT == "production":
    cors_allow_origins = PRODUCTION_WEB_ORIGINS.copy()
else:
    cors_allow_origins = PRODUCTION_WEB_ORIGINS + DEVELOPMENT_WEB_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Idempotency-Key",
        "X-Request-Id",
    ],
    max_age=600,
)


def apply_security_headers(response: Response, request_id: str, is_secure: bool) -> Response:
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    if ENABLE_HSTS and is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    if request.headers.get("x-user-id") and request.url.path in PROTECTED_PATHS:
        internal_auth = request.headers.get("x-internal-auth")
        if not INTERNAL_GATEWAY_SECRET or internal_auth != INTERNAL_GATEWAY_SECRET:
            response = JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "X-User-Id header is not allowed on public requests"},
            )
            return apply_security_headers(response, request_id, request.url.scheme == "https")

    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
        user_id = await verify_bearer_token(token)
        if user_id:
            request.state.user_id = user_id

    response = await call_next(request)
    return apply_security_headers(response, request_id, request.url.scheme == "https")


async def read_body_with_limit(request: Request, limit: int) -> Optional[bytes]:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    content_length = request.headers.get("content-length")
    size_known = False
    if content_length:
        try:
            size = int(content_length)
            size_known = True
        except ValueError:
            size = None
        if size_known and size > MAX_REQUEST_BYTES:
            logger.warning(
                "payload_too_large path=%s ip=%s size=%s limit=%s request_id=%s",
                request.url.path,
                get_client_ip(request),
                size,
                MAX_REQUEST_BYTES,
                get_request_id(request),
            )
            return JSONResponse(status_code=413, content={"detail": "Payload Too Large"})
    if request.method in {"POST", "PUT", "PATCH"} and not size_known:
        body = await read_body_with_limit(request, MAX_REQUEST_BYTES)
        if body is None:
            logger.warning(
                "payload_too_large path=%s ip=%s size=unknown limit=%s request_id=%s",
                request.url.path,
                get_client_ip(request),
                MAX_REQUEST_BYTES,
                get_request_id(request),
            )
            return JSONResponse(status_code=413, content={"detail": "Payload Too Large"})
        request._body = body
    return await call_next(request)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Category = Literal["politics", "sports", "finance", "entertainment"]
Status = Literal["open", "closed", "settled"]


class Market(BaseModel):
    id: str
    title: str
    status: Status
    yesPrice: float
    noPrice: float
    category: Category
    volumeKr: int
    description: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: str


class BetRequest(BaseModel):
    market_id: str
    outcome: Literal["yes", "no"]
    stake_dkk: int = Field(..., ge=1)


class PayoutRequest(BaseModel):
    bet_id: int
    amount_dkk: int = Field(..., ge=1)


DEMO_MARKETS: List[Market] = [
    Market(
        id="novo_1000",
        title="Vil Novo Nordisk aktien overstige 1000 DKK?",
        status="settled",
        yesPrice=0.52,
        noPrice=0.48,
        category="finance",
        volumeKr=5_700_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="vingegaard_tdf_2025",
        title="Vil Jonas Vingegaard vinde Tour de France 2025?",
        status="settled",
        yesPrice=0.38,
        noPrice=0.62,
        category="sports",
        volumeKr=4_100_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="dk_vm_2026",
        title="Vil Danmark kvalificere sig til VM 2026?",
        status="settled",
        yesPrice=0.82,
        noPrice=0.18,
        category="sports",
        volumeKr=3_500_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="fck_superliga_2425",
        title="Vil FC Koebenhavn vinde Superligaen 2024/25?",
        status="settled",
        yesPrice=0.45,
        noPrice=0.55,
        category="sports",
        volumeKr=2_300_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="c25_3000_2025",
        title="Vil C25 indekset naa 3000 point i 2025?",
        status="settled",
        yesPrice=0.48,
        noPrice=0.52,
        category="finance",
        volumeKr=2_300_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="dk_eurovision_2025",
        title="Vil Danmark vinde Eurovision 2025?",
        status="settled",
        yesPrice=0.12,
        noPrice=0.88,
        category="entertainment",
        volumeKr=2_300_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="dk_handbold_vm_2025",
        title="Vil det danske haandboldlandshold vinde VM 2025?",
        status="settled",
        yesPrice=0.28,
        noPrice=0.72,
        category="sports",
        volumeKr=1_900_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="vestas_20_2025",
        title="Vil Vestas aktien stige 20% i 2025?",
        status="settled",
        yesPrice=0.35,
        noPrice=0.65,
        category="finance",
        volumeKr=1_900_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="mf_statsminister_2025",
        title="Vil Mette Frederiksen forblive statsminister hele 2025?",
        status="settled",
        yesPrice=0.75,
        noPrice=0.25,
        category="politics",
        volumeKr=1_600_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="maersk_15000_2025",
        title="Vil Maersk aktien vaere over 15.000 DKK ved aarets udgang?",
        status="settled",
        yesPrice=0.41,
        noPrice=0.59,
        category="finance",
        volumeKr=1_400_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="s_vinder_valg",
        title="Vil Socialdemokratiet vinde naeste folketingsvalg?",
        status="settled",
        yesPrice=0.44,
        noPrice=0.56,
        category="politics",
        volumeKr=1_300_000,
        description="Demo market for UI wiring",
    ),
    Market(
        id="dk_saenke_skat_2025",
        title="Vil Danmark saenke skatten i 2025?",
        status="settled",
        yesPrice=0.31,
        noPrice=0.69,
        category="politics",
        volumeKr=1_200_000,
        description="Demo market for UI wiring",
    ),
]


@app.get("/", dependencies=[limiter_dep(LIMITS_READ)])
def root():
    return {"message": "Backend is running", "docs": "/docs", "health": "/health", "markets": "/markets"}


@app.get("/health", dependencies=[limiter_dep(LIMITS_READ)])
def health():
    return {"status": "ok"}


@app.get("/markets", response_model=List[Market], dependencies=[limiter_dep(LIMITS_READ)])
def list_markets(request: Request):
    started_at = time.monotonic()
    markets = DEMO_MARKETS
    logger.info(
        "markets_list_ok request_id=%s count=%s duration_ms=%s",
        get_request_id(request),
        len(markets),
        int((time.monotonic() - started_at) * 1000),
    )
    return markets


@app.get("/markets/{market_id}", response_model=Market, dependencies=[limiter_dep(LIMITS_READ)])
def get_market(market_id: str):
    for market in DEMO_MARKETS:
        if market.id == market_id:
            return market
    raise HTTPException(status_code=404, detail="Market not found")


@app.post("/auth/login", dependencies=[limiter_dep(LIMITS_AUTH)])
def login(payload: LoginRequest):
    if not payload.email or not payload.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    user_id = "demo-user"
    return {
        "user_id": user_id,
        "access_token": sign_user_id(user_id),
        "token_type": "bearer",
        "redirect": "/markets",
    }


@app.post("/auth/signup", dependencies=[limiter_dep(LIMITS_AUTH)])
def signup(payload: SignupRequest):
    if not payload.email or not payload.password:
        raise HTTPException(status_code=400, detail="Invalid signup data")
    user_id = "demo-user"
    return {
        "user_id": user_id,
        "access_token": sign_user_id(user_id),
        "token_type": "bearer",
        "redirect": "/signin",
    }


@app.post("/auth/forgot-password", dependencies=[limiter_dep(LIMITS_AUTH)])
def forgot_password(payload: ForgotPasswordRequest):
    if not payload.email:
        raise HTTPException(status_code=400, detail="Email required")
    return {"message": "Reset link sent"}


async def fetch_quote_for_market(market: Market) -> dict:
    return {"market_id": market.id, "yes_price": market.yesPrice, "no_price": market.noPrice}


@app.get("/odds/quote", dependencies=[limiter_dep(LIMITS_AUTH)])
async def odds_quote(market_id: str):
    for market in DEMO_MARKETS:
        if market.id == market_id:
            if not odds_breaker.allow():
                raise HTTPException(status_code=503, detail="Odds service temporarily unavailable")
            try:
                response = await asyncio.wait_for(
                    fetch_quote_for_market(market),
                    timeout=EXTERNAL_REQUEST_TIMEOUT_SECONDS,
                )
                odds_breaker.record_success()
                return response
            except asyncio.TimeoutError as exc:
                odds_breaker.record_failure()
                raise HTTPException(status_code=504, detail="Upstream timeout") from exc
            except Exception as exc:
                odds_breaker.record_failure()
                raise HTTPException(status_code=503, detail="Odds service error") from exc
    raise HTTPException(status_code=404, detail="Market not found")


def fetch_idempotency_record(
    db: Session,
    user_id: str,
    key: str,
    endpoint: str,
) -> Optional[IdempotencyRecord]:
    record = (
        db.query(IdempotencyRecord)
        .execution_options(populate_existing=True)
        .filter(
            IdempotencyRecord.user_id == user_id,
            IdempotencyRecord.key == key,
            IdempotencyRecord.endpoint == endpoint,
        )
        .one_or_none()
    )
    if record and record.expires_at < datetime.utcnow():
        db.delete(record)
        db.commit()
        return None
    return record


def store_idempotency_response(
    db: Session,
    record: IdempotencyRecord,
    status_code: int,
    body: dict,
) -> None:
    if status_code >= 500:
        return
    record.status_code = status_code
    record.response_body = json.dumps(body)
    db.add(record)
    db.commit()


def clear_idempotency_record(db: Session, record: IdempotencyRecord) -> None:
    db.delete(record)
    db.commit()


def wait_for_idempotency_response(
    db: Session,
    user_id: str,
    key: str,
    endpoint: str,
) -> Optional[JSONResponse]:
    deadline = time.monotonic() + IDEMPOTENCY_WAIT_SECONDS
    while time.monotonic() < deadline:
        record = fetch_idempotency_record(db, user_id, key, endpoint)
        if record and record.status_code and record.response_body:
            return JSONResponse(
                status_code=record.status_code,
                content=json.loads(record.response_body),
            )
        time.sleep(0.05)
    return None


@app.post("/bets", dependencies=[limiter_dep(LIMITS_BETS)])
def place_bet(
    payload: BetRequest,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    user_id: str = Depends(require_user_id),
    db: Session = Depends(get_db),
):
    endpoint = request.url.path
    request_hash = hash_payload(payload.model_dump())
    existing = fetch_idempotency_record(db, user_id, idempotency_key, endpoint)

    if existing:
        if existing.request_hash != request_hash:
            audit_logger.warning(
                "bet_reject reason=idempotency_mismatch user_id=%s key=%s ip=%s ua=%s request_id=%s",
                user_id,
                idempotency_key,
                get_client_ip(request),
                request.headers.get("user-agent", "-"),
                get_request_id(request),
            )
            raise HTTPException(status_code=409, detail="Idempotency key reuse with different payload")
        if existing.status_code and existing.response_body:
            audit_logger.info(
                "bet_idempotent_hit user_id=%s key=%s ip=%s ua=%s request_id=%s",
                user_id,
                idempotency_key,
                get_client_ip(request),
                request.headers.get("user-agent", "-"),
                get_request_id(request),
            )
            return JSONResponse(
                status_code=existing.status_code,
                content=json.loads(existing.response_body),
            )
        waited = wait_for_idempotency_response(db, user_id, idempotency_key, endpoint)
        if waited:
            return waited
        raise HTTPException(status_code=409, detail="Request in progress")

    record = IdempotencyRecord(
        user_id=user_id,
        key=idempotency_key,
        endpoint=endpoint,
        request_hash=request_hash,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=IDEMPOTENCY_TTL_HOURS),
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = fetch_idempotency_record(db, user_id, idempotency_key, endpoint)
        if existing and existing.response_body and existing.status_code:
            return JSONResponse(
                status_code=existing.status_code,
                content=json.loads(existing.response_body),
            )
        waited = wait_for_idempotency_response(db, user_id, idempotency_key, endpoint)
        if waited:
            return waited
        raise HTTPException(status_code=409, detail="Request in progress")

    try:
        if not any(market.id == payload.market_id for market in DEMO_MARKETS):
            response_body = {"detail": "Market not found"}
            store_idempotency_response(db, record, status.HTTP_404_NOT_FOUND, response_body)
            audit_logger.warning(
                "bet_reject reason=market_not_found user_id=%s key=%s ip=%s ua=%s request_id=%s",
                user_id,
                idempotency_key,
                get_client_ip(request),
                request.headers.get("user-agent", "-"),
                get_request_id(request),
            )
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=response_body)
        bet = BetRecord(
            user_id=user_id,
            market_id=payload.market_id,
            outcome=payload.outcome,
            stake_dkk=payload.stake_dkk,
        )
        db.add(bet)
        db.flush()

        response_body = {
            "bet_id": bet.id,
            "status": "accepted",
            "market_id": payload.market_id,
            "outcome": payload.outcome,
            "stake_dkk": payload.stake_dkk,
        }
        store_idempotency_response(db, record, status.HTTP_201_CREATED, response_body)
        audit_logger.info(
            "bet_accept user_id=%s bet_id=%s key=%s ip=%s ua=%s request_id=%s",
            user_id,
            bet.id,
            idempotency_key,
            get_client_ip(request),
            request.headers.get("user-agent", "-"),
            get_request_id(request),
        )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=response_body)
    except Exception as exc:
        clear_idempotency_record(db, record)
        audit_logger.warning(
            "bet_reject reason=exception user_id=%s key=%s ip=%s ua=%s request_id=%s",
            user_id,
            idempotency_key,
            get_client_ip(request),
            request.headers.get("user-agent", "-"),
            get_request_id(request),
        )
        raise HTTPException(status_code=500, detail="Bet placement failed") from exc


@app.post("/payouts", dependencies=[limiter_dep(LIMITS_PAYOUTS)])
def payout(
    payload: PayoutRequest,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    user_id: str = Depends(require_user_id),
    db: Session = Depends(get_db),
):
    endpoint = request.url.path
    request_hash = hash_payload(payload.model_dump())
    existing = fetch_idempotency_record(db, user_id, idempotency_key, endpoint)

    if existing:
        if existing.request_hash != request_hash:
            raise HTTPException(status_code=409, detail="Idempotency key reuse with different payload")
        if existing.status_code and existing.response_body:
            return JSONResponse(
                status_code=existing.status_code,
                content=json.loads(existing.response_body),
            )
        waited = wait_for_idempotency_response(db, user_id, idempotency_key, endpoint)
        if waited:
            return waited
        raise HTTPException(status_code=409, detail="Request in progress")

    record = IdempotencyRecord(
        user_id=user_id,
        key=idempotency_key,
        endpoint=endpoint,
        request_hash=request_hash,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=IDEMPOTENCY_TTL_HOURS),
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = fetch_idempotency_record(db, user_id, idempotency_key, endpoint)
        if existing and existing.response_body and existing.status_code:
            return JSONResponse(
                status_code=existing.status_code,
                content=json.loads(existing.response_body),
            )
        waited = wait_for_idempotency_response(db, user_id, idempotency_key, endpoint)
        if waited:
            return waited
        raise HTTPException(status_code=409, detail="Request in progress")

    try:
        bet_exists = db.query(BetRecord).filter(BetRecord.id == payload.bet_id).first()
        if not bet_exists:
            response_body = {"detail": "Bet not found"}
            store_idempotency_response(db, record, status.HTTP_404_NOT_FOUND, response_body)
            audit_logger.warning(
                "payout_reject reason=bet_not_found user_id=%s key=%s ip=%s ua=%s request_id=%s",
                user_id,
                idempotency_key,
                get_client_ip(request),
                request.headers.get("user-agent", "-"),
                get_request_id(request),
            )
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=response_body)

        response_body = {
            "payout_id": f"payout-{payload.bet_id}",
            "status": "accepted",
            "bet_id": payload.bet_id,
            "amount_dkk": payload.amount_dkk,
        }
        store_idempotency_response(db, record, status.HTTP_201_CREATED, response_body)
        audit_logger.info(
            "payout_accept user_id=%s bet_id=%s key=%s ip=%s ua=%s request_id=%s",
            user_id,
            payload.bet_id,
            idempotency_key,
            get_client_ip(request),
            request.headers.get("user-agent", "-"),
            get_request_id(request),
        )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=response_body)
    except Exception as exc:
        clear_idempotency_record(db, record)
        audit_logger.warning(
            "payout_reject reason=exception user_id=%s key=%s ip=%s ua=%s request_id=%s",
            user_id,
            idempotency_key,
            get_client_ip(request),
            request.headers.get("user-agent", "-"),
            get_request_id(request),
        )
        raise HTTPException(status_code=500, detail="Payout failed") from exc
