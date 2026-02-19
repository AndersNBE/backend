import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import os
import ssl
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Literal, Optional
from urllib.parse import urlparse, urlunparse

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

from .routers.trading import router as trading_router
from .storage import (
    BetRecord,
    IdempotencyRecord,
    TradingMarket,
    TradingMarketStatus,
    get_db,
    init_db,
)

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
REDIS_INIT_TIMEOUT_SECONDS = float(os.getenv("REDIS_INIT_TIMEOUT_SECONDS", "2.5"))
RATE_LIMITER_RETRY_LOOP_SECONDS = float(os.getenv("RATE_LIMITER_RETRY_LOOP_SECONDS", "10"))
RATE_LIMITER_STARTUP_WAIT_SECONDS = float(os.getenv("RATE_LIMITER_STARTUP_WAIT_SECONDS", "0.25"))
RATE_LIMITER_FAIL_CLOSED = os.getenv("RATE_LIMITER_FAIL_CLOSED", "false").lower() == "true"
REDIS_SSL_CERT_REQS = os.getenv("REDIS_SSL_CERT_REQS", "required").lower()
REDIS_FORCE_TLS = os.getenv("REDIS_FORCE_TLS", "false").lower() == "true"

if not SUPABASE_JWT_ISSUER and SUPABASE_URL:
    SUPABASE_JWT_ISSUER = f"{SUPABASE_URL.rstrip('/')}/auth/v1"
SUPABASE_JWKS_URL = (
    f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json" if SUPABASE_URL else None
)
SUPABASE_ALLOWED_ALGORITHMS = {"RS256", "ES256", "EdDSA"}


def _build_supabase_jwk_client():
    if not SUPABASE_JWKS_URL:
        return None

    kwargs = {}
    try:
        signature = inspect.signature(jwt.PyJWKClient.__init__)
        if "lifespan" in signature.parameters:
            kwargs["lifespan"] = SUPABASE_JWKS_CACHE_TTL_SECONDS
        if "timeout" in signature.parameters:
            kwargs["timeout"] = SUPABASE_JWKS_REQUEST_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        kwargs = {}

    return jwt.PyJWKClient(SUPABASE_JWKS_URL, **kwargs)


SUPABASE_JWK_CLIENT = _build_supabase_jwk_client()

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


def looks_like_jwt(token: str) -> bool:
    parts = token.split(".")
    return len(parts) == 3 and all(parts)


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
    if SUPABASE_JWK_CLIENT and SUPABASE_JWT_ISSUER:
        headers = jwt.get_unverified_header(token)
        algorithm = headers.get("alg")
        if not isinstance(algorithm, str) or algorithm not in SUPABASE_ALLOWED_ALGORITHMS:
            raise jwt.InvalidAlgorithmError("Unsupported JWT signing algorithm.")
        signing_key = await asyncio.to_thread(SUPABASE_JWK_CLIENT.get_signing_key_from_jwt, token)

        def _decode_claims() -> dict:
            if SUPABASE_JWT_AUDIENCE:
                return jwt.decode(
                    token,
                    key=signing_key.key,
                    algorithms=[algorithm],
                    audience=SUPABASE_JWT_AUDIENCE,
                    issuer=SUPABASE_JWT_ISSUER,
                    options={"require": ["sub", "exp"]},
                )
            return jwt.decode(
                token,
                key=signing_key.key,
                algorithms=[algorithm],
                issuer=SUPABASE_JWT_ISSUER,
                options={"verify_aud": False, "require": ["sub", "exp"]},
            )

        claims = await asyncio.to_thread(_decode_claims)

        subject = claims.get("sub")
        if isinstance(subject, str) and subject:
            return subject
        raise jwt.InvalidTokenError("JWT missing 'sub' claim.")

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
rate_limiter_retry_task: Optional[asyncio.Task] = None
rate_limiter_retry_stop_event: Optional[asyncio.Event] = None
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


def _redis_endpoint_for_logs(redis_url: str) -> str:
    try:
        parsed = urlparse(redis_url)
        return f"{parsed.scheme}://{parsed.hostname or '-'}:{parsed.port or '-'}"
    except Exception:
        return "<invalid-redis-url>"


def _redis_ssl_cert_reqs_mode() -> int:
    mapping = {
        "none": ssl.CERT_NONE,
        "optional": ssl.CERT_OPTIONAL,
        "required": ssl.CERT_REQUIRED,
    }
    if REDIS_SSL_CERT_REQS not in mapping:
        raise ValueError(
            f"Invalid REDIS_SSL_CERT_REQS='{REDIS_SSL_CERT_REQS}'. Use one of: none, optional, required."
        )
    return mapping[REDIS_SSL_CERT_REQS]


def _normalize_redis_url(redis_url: str) -> str:
    parsed = urlparse(redis_url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"redis", "rediss", "fakeredis"}:
        raise ValueError(
            f"Unsupported Redis URL scheme '{parsed.scheme}'. Use redis:// or rediss://."
        )

    if REDIS_FORCE_TLS and scheme == "redis":
        parsed = parsed._replace(scheme="rediss")

    return urlunparse(parsed)


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
    normalized_url = _normalize_redis_url(redis_url)
    parsed = urlparse(normalized_url)
    scheme = parsed.scheme.lower()

    if scheme == "fakeredis":
        from fakeredis.aioredis import FakeRedis

        return FakeRedis(decode_responses=True)

    client_kwargs = {
        "encoding": "utf-8",
        "decode_responses": True,
        "socket_connect_timeout": REDIS_CONNECT_TIMEOUT_SECONDS,
        "socket_timeout": REDIS_SOCKET_TIMEOUT_SECONDS,
        "health_check_interval": REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        "retry_on_timeout": True,
        "socket_keepalive": True,
    }
    if scheme == "rediss":
        client_kwargs["ssl"] = True
        client_kwargs["ssl_cert_reqs"] = _redis_ssl_cert_reqs_mode()

    return redis.from_url(normalized_url, **client_kwargs)


async def _close_redis_client_instance(client: Optional[redis.Redis], reason: str) -> None:
    if client is None:
        return

    close_method = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close_method is None:
        return

    try:
        result = close_method()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("rate_limiter_redis_close_failed reason=%s", reason)


async def _close_rate_limiter_client() -> None:
    global rate_limiter_client
    client = rate_limiter_client
    rate_limiter_client = None
    await _close_redis_client_instance(client, reason="global_close")


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
            "rate_limiter_unavailable reason=%s path=%s redis=%s retry_in=%.1fs local_fallback=enabled error_type=%s error=%s",
            reason,
            request_path,
            _redis_endpoint_for_logs(REDIS_URL) if REDIS_URL else "-",
            RATE_LIMITER_RETRY_SECONDS,
            error.__class__.__name__,
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

        client: Optional[redis.Redis] = None
        try:
            client = _build_redis_client(REDIS_URL)
            await asyncio.wait_for(client.ping(), timeout=REDIS_INIT_TIMEOUT_SECONDS)
            await asyncio.wait_for(
                FastAPILimiter.init(
                    client,
                    identifier=rate_limit_identifier,
                    http_callback=rate_limit_callback,
                ),
                timeout=REDIS_INIT_TIMEOUT_SECONDS,
            )
            rate_limiter_client = client
            rate_limiter_ready = True
            rate_limiter_next_retry_at = 0.0
            logger.info(
                "rate_limiter_initialized redis=%s",
                _redis_endpoint_for_logs(REDIS_URL),
            )
            return True
        except Exception as exc:
            await _close_redis_client_instance(client, reason="init_failed")
            await _mark_rate_limiter_unavailable("init_failed", exc)
            return False


async def _rate_limiter_retry_loop() -> None:
    global rate_limiter_retry_stop_event
    if rate_limiter_retry_stop_event is None:
        rate_limiter_retry_stop_event = asyncio.Event()

    logger.info(
        "rate_limiter_retry_loop_started interval_seconds=%.1f",
        RATE_LIMITER_RETRY_LOOP_SECONDS,
    )

    while not rate_limiter_retry_stop_event.is_set():
        if ENABLE_RATE_LIMITING and not rate_limiter_ready:
            await _initialize_rate_limiter()

        try:
            await asyncio.wait_for(
                rate_limiter_retry_stop_event.wait(),
                timeout=RATE_LIMITER_RETRY_LOOP_SECONDS,
            )
        except asyncio.TimeoutError:
            continue

    logger.info("rate_limiter_retry_loop_stopped")


def _start_rate_limiter_retry_loop() -> None:
    global rate_limiter_retry_task, rate_limiter_retry_stop_event
    if not ENABLE_RATE_LIMITING:
        return

    if rate_limiter_retry_stop_event is None:
        rate_limiter_retry_stop_event = asyncio.Event()

    if rate_limiter_retry_task and not rate_limiter_retry_task.done():
        return

    rate_limiter_retry_task = asyncio.create_task(_rate_limiter_retry_loop())


async def _stop_rate_limiter_retry_loop() -> None:
    global rate_limiter_retry_task, rate_limiter_retry_stop_event

    if rate_limiter_retry_stop_event is not None:
        rate_limiter_retry_stop_event.set()

    if rate_limiter_retry_task is not None:
        try:
            await rate_limiter_retry_task
        except Exception:
            logger.exception("rate_limiter_retry_loop_failed")

    rate_limiter_retry_task = None
    rate_limiter_retry_stop_event = None


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
    else:
        try:
            normalized_redis_url = _normalize_redis_url(REDIS_URL)
            redis_scheme = urlparse(normalized_redis_url).scheme.lower()
            logger.info(
                "rate_limiter_config redis=%s scheme=%s init_timeout=%.1fs retry_loop=%.1fs fail_closed=%s",
                _redis_endpoint_for_logs(normalized_redis_url),
                redis_scheme,
                REDIS_INIT_TIMEOUT_SECONDS,
                RATE_LIMITER_RETRY_LOOP_SECONDS,
                RATE_LIMITER_FAIL_CLOSED,
            )
            if ENVIRONMENT == "production" and redis_scheme == "redis":
                logger.warning(
                    "rate_limiter_non_tls_scheme redis=%s. If Railway requires TLS for this endpoint, switch REDIS_URL to rediss://",
                    _redis_endpoint_for_logs(normalized_redis_url),
                )
        except Exception as exc:
            message = f"Invalid REDIS_URL configuration: {exc}"
            if ENVIRONMENT == "production":
                raise RuntimeError(message) from exc
            logger.error(message)

    if ENVIRONMENT == "production" and not SUPABASE_URL:
        raise RuntimeError("Missing SUPABASE_URL. Set it to verify Supabase access tokens.")
    if not SUPABASE_URL and AUTH_SECRET:
        logger.warning("SUPABASE_URL missing; using legacy AUTH_SECRET bearer verification.")

    if ENABLE_RATE_LIMITING:
        _start_rate_limiter_retry_loop()
        init_task = asyncio.create_task(_initialize_rate_limiter(force=True))
        initialized = False
        try:
            initialized = await asyncio.wait_for(
                asyncio.shield(init_task),
                timeout=RATE_LIMITER_STARTUP_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "rate_limiter_startup_probe_timeout wait_seconds=%.2f. Startup continues with local fallback until Redis reconnects.",
                RATE_LIMITER_STARTUP_WAIT_SECONDS,
            )
        except Exception as exc:
            await _mark_rate_limiter_unavailable("startup_probe_failed", exc)

        if not initialized:
            message = (
                "rate_limiter_startup_degraded: Redis not ready. "
                "Requests use local fallback and background retries remain active."
            )
            if ENVIRONMENT == "production" and RATE_LIMITER_FAIL_CLOSED:
                raise RuntimeError(message)
            logger.error(message)

    init_db()
    try:
        yield
    finally:
        await _stop_rate_limiter_retry_loop()
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
app.include_router(trading_router)


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
    request.state.user_id = None

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
        if looks_like_jwt(token):
            try:
                user_id = await verify_bearer_token(token)
            except Exception as exc:
                logger.info(
                    "auth_token_invalid path=%s request_id=%s error_type=%s",
                    request.url.path,
                    request_id,
                    exc.__class__.__name__,
                )
                user_id = None
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


TRADING_TO_LEGACY_STATUS: dict[str, Status] = {
    TradingMarketStatus.draft.value: "open",
    TradingMarketStatus.review.value: "open",
    TradingMarketStatus.scheduled.value: "open",
    TradingMarketStatus.live.value: "open",
    TradingMarketStatus.trading_paused.value: "open",
    TradingMarketStatus.closed.value: "closed",
    TradingMarketStatus.resolving.value: "closed",
    TradingMarketStatus.settled.value: "settled",
    TradingMarketStatus.cancelled.value: "closed",
}


def _enum_value(raw: object) -> str:
    value = getattr(raw, "value", None)
    if isinstance(value, str):
        return value
    return str(raw)


def _market_category(raw: object) -> Category:
    category = _enum_value(raw)
    if category in {"politics", "sports", "finance", "entertainment"}:
        return category  # type: ignore[return-value]
    return "finance"


def _market_status(raw: object) -> Status:
    return TRADING_TO_LEGACY_STATUS.get(_enum_value(raw), "open")


def to_market_response(market: TradingMarket) -> Market:
    yes_price_bps = int(market.yes_price_bps)
    no_price_bps = int(getattr(market, "no_price_bps", 10000 - yes_price_bps))
    return Market(
        id=market.slug,
        title=market.title,
        status=_market_status(market.status),
        yesPrice=max(0, min(1, yes_price_bps / 10000)),
        noPrice=max(0, min(1, no_price_bps / 10000)),
        category=_market_category(market.category),
        volumeKr=0,
        description=market.description,
    )


def fetch_market_by_slug(db: Session, market_slug: str) -> Optional[TradingMarket]:
    return (
        db.query(TradingMarket)
        .filter(TradingMarket.slug == market_slug)
        .one_or_none()
    )


@app.get("/", dependencies=[limiter_dep(LIMITS_READ)])
def root():
    return {
        "message": "Backend is running",
        "docs": "/docs",
        "health": "/health",
        "markets": "/markets",
        "trading_markets": "/trading/markets",
    }


@app.get("/health", dependencies=[limiter_dep(LIMITS_READ)])
def health():
    return {
        "status": "ok",
        "rate_limiter": {
            "enabled": ENABLE_RATE_LIMITING,
            "ready": rate_limiter_ready,
            "degraded": bool(ENABLE_RATE_LIMITING and not rate_limiter_ready),
            "redis": _redis_endpoint_for_logs(REDIS_URL) if REDIS_URL else None,
        },
    }


@app.get("/markets", response_model=List[Market], dependencies=[limiter_dep(LIMITS_READ)])
def list_markets(request: Request, db: Session = Depends(get_db)):
    started_at = time.monotonic()
    rows = (
        db.query(TradingMarket)
        .order_by(TradingMarket.created_at.desc())
        .all()
    )
    markets = [to_market_response(row) for row in rows]
    logger.info(
        "markets_list_ok request_id=%s count=%s duration_ms=%s",
        get_request_id(request),
        len(markets),
        int((time.monotonic() - started_at) * 1000),
    )
    return markets


@app.get("/markets/{market_id}", response_model=Market, dependencies=[limiter_dep(LIMITS_READ)])
def get_market(market_id: str, db: Session = Depends(get_db)):
    market = fetch_market_by_slug(db, market_id)
    if market is not None:
        return to_market_response(market)
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
async def odds_quote(market_id: str, db: Session = Depends(get_db)):
    market_row = fetch_market_by_slug(db, market_id)
    if market_row is None:
        raise HTTPException(status_code=404, detail="Market not found")

    market = to_market_response(market_row)
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
        if fetch_market_by_slug(db, payload.market_id) is None:
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
