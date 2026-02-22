import enum
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..storage import (
    TradingMarket,
    TradingMarketCategory,
    TradingMarketRule,
    TradingMarketStateEvent,
    TradingMarketStatus,
    get_db,
)
from ..trading_validation import (
    CreateAdminMarketRequest,
    TradingCategory,
    TradingStatus,
    UpdateAdminMarketRequest,
    validate_market_times,
)

router = APIRouter(prefix="/trading", tags=["trading"])

VISIBLE_STATUSES = (
    TradingMarketStatus.live,
    TradingMarketStatus.trading_paused,
    TradingMarketStatus.closed,
    TradingMarketStatus.resolving,
    TradingMarketStatus.settled,
    TradingMarketStatus.cancelled,
)


class TradingMarketResponse(BaseModel):
    id: str
    publicId: str
    slug: str
    title: str
    description: str | None = None
    category: TradingCategory
    status: TradingStatus
    yesPriceBps: int
    noPriceBps: int
    yesPrice: float
    noPrice: float
    tickSizeBps: int
    minOrderSizeMinor: int
    maxOrderSizeMinor: int
    openTime: datetime
    closeTime: datetime
    resolveTime: datetime | None = None
    tradingEndsAt: datetime
    resolvesAfter: datetime
    createdAt: datetime
    updatedAt: datetime


def _enum_value(value: object) -> str:
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _parse_optional_uuid(raw_value: str | None) -> uuid.UUID | None:
    if not raw_value:
        return None
    try:
        return uuid.UUID(raw_value)
    except (ValueError, TypeError):
        return None


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_times(open_time: datetime, close_time: datetime, resolve_time: datetime | None) -> None:
    try:
        validate_market_times(open_time, close_time, resolve_time)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _rules_to_text(rules_json: dict[str, Any]) -> str:
    return json.dumps(rules_json, sort_keys=True, separators=(",", ":"))


def _serialize_market(market: TradingMarket) -> TradingMarketResponse:
    yes_price_bps = int(market.yes_price_bps)
    no_price_bps = int(getattr(market, "no_price_bps", 10000 - yes_price_bps))
    category = _enum_value(market.category)
    status_value = _enum_value(market.status)
    open_time = market.open_time or market.trading_starts_at
    close_time = market.close_time or market.trading_ends_at
    resolve_time = market.resolve_time
    resolves_after = market.resolves_after or close_time

    if open_time is None or close_time is None:
        raise HTTPException(status_code=500, detail="Market has invalid schedule timestamps")

    return TradingMarketResponse(
        id=str(market.public_id),
        publicId=str(market.public_id),
        slug=market.slug,
        title=market.title,
        description=market.description,
        category=category,  # type: ignore[arg-type]
        status=status_value,  # type: ignore[arg-type]
        yesPriceBps=yes_price_bps,
        noPriceBps=no_price_bps,
        yesPrice=round(yes_price_bps / 10000, 5),
        noPrice=round(no_price_bps / 10000, 5),
        tickSizeBps=int(market.tick_size_bps),
        minOrderSizeMinor=int(market.min_order_size_minor),
        maxOrderSizeMinor=int(market.max_order_size_minor),
        openTime=open_time,
        closeTime=close_time,
        resolveTime=resolve_time,
        tradingEndsAt=close_time,
        resolvesAfter=resolves_after,
        createdAt=market.created_at,
        updatedAt=market.updated_at,
    )


def _require_admin_key(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> None:
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_API_KEY is not configured",
        )
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _get_active_market_rule(db: Session, market_id: int) -> TradingMarketRule | None:
    return (
        db.query(TradingMarketRule)
        .filter(
            TradingMarketRule.market_id == market_id,
            TradingMarketRule.is_active.is_(True),
        )
        .order_by(TradingMarketRule.version.desc())
        .first()
    )


def _error_response_from_integrity(exc: IntegrityError) -> HTTPException:
    message = str(exc.orig)
    if "markets_slug_uniq" in message:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Market slug already exists")
    if "markets_slug_format_chk" in message:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid slug format")
    if "markets_open_before_close_chk" in message:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="open_time must be earlier than close_time")
    if "markets_resolve_after_close_chk" in message:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="resolve_time must be greater than or equal to close_time",
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Database integrity error",
    )


@router.get("/markets", response_model=list[TradingMarketResponse])
def list_trading_markets(db: Session = Depends(get_db)):
    markets = (
        db.query(TradingMarket)
        .filter(TradingMarket.status.in_(VISIBLE_STATUSES))
        .order_by(TradingMarket.close_time.asc(), TradingMarket.created_at.desc())
        .all()
    )
    return [_serialize_market(market) for market in markets]


@router.get("/markets/{slug}", response_model=TradingMarketResponse)
def get_trading_market_by_slug(slug: str, db: Session = Depends(get_db)):
    market = (
        db.query(TradingMarket)
        .filter(
            TradingMarket.slug == slug,
            TradingMarket.status.in_(VISIBLE_STATUSES),
        )
        .one_or_none()
    )
    if market is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Market not found")
    return _serialize_market(market)


@router.post(
    "/admin/markets",
    response_model=TradingMarketResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_require_admin_key)],
)
def create_admin_market(
    payload: CreateAdminMarketRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    actor_id = _parse_optional_uuid(getattr(request.state, "user_id", None))
    now = datetime.now(timezone.utc)
    open_time = _to_utc(payload.open_time)
    close_time = _to_utc(payload.close_time)
    resolve_time = _to_utc(payload.resolve_time) if payload.resolve_time else None
    _validate_times(open_time, close_time, resolve_time)

    market = TradingMarket(
        slug=payload.slug,
        title=payload.title,
        description=payload.description,
        category=TradingMarketCategory(payload.category),
        status=TradingMarketStatus.draft,
        yes_price_bps=payload.yes_price_bps,
        tick_size_bps=payload.tick_size_bps,
        min_order_size_minor=payload.min_order_size_minor,
        max_order_size_minor=payload.max_order_size_minor,
        trading_starts_at=open_time,
        trading_ends_at=close_time,
        resolves_after=resolve_time or close_time,
        open_time=open_time,
        close_time=close_time,
        resolve_time=resolve_time,
        created_by=actor_id,
        updated_by=actor_id,
        created_at=now,
        updated_at=now,
    )

    db.add(market)
    try:
        db.flush()
        db.add(
            TradingMarketRule(
                market_id=market.id,
                version=1,
                rule_text=_rules_to_text(payload.rules_json),
                rules_json=payload.rules_json,
                is_active=True,
                created_by=actor_id,
                created_at=now,
                updated_at=now,
            )
        )
        db.add(
            TradingMarketStateEvent(
                market_id=market.id,
                from_status=None,
                to_status=TradingMarketStatus.draft,
                reason="created",
                event_metadata={"event": "created", "source": "admin_api_key"},
                changed_by=actor_id,
                created_at=now,
            )
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise _error_response_from_integrity(exc) from exc

    db.refresh(market)
    return _serialize_market(market)


@router.patch(
    "/admin/markets/{market_id}",
    response_model=TradingMarketResponse,
    dependencies=[Depends(_require_admin_key)],
)
def patch_admin_market(
    market_id: str,
    payload: UpdateAdminMarketRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        market_public_id = uuid.UUID(market_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid market_id") from exc

    market = (
        db.query(TradingMarket)
        .filter(TradingMarket.public_id == market_public_id)
        .one_or_none()
    )
    if market is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Market not found")

    updates = payload.model_dump(exclude_unset=True)
    actor_id = _parse_optional_uuid(getattr(request.state, "user_id", None))
    now = datetime.now(timezone.utc)

    current_open = market.open_time or market.trading_starts_at
    current_close = market.close_time or market.trading_ends_at
    current_resolve = market.resolve_time
    if current_open is None or current_close is None:
        raise HTTPException(status_code=500, detail="Market has invalid schedule timestamps")

    next_open = _to_utc(updates["open_time"]) if "open_time" in updates and updates["open_time"] else current_open
    next_close = _to_utc(updates["close_time"]) if "close_time" in updates and updates["close_time"] else current_close
    if "resolve_time" in updates:
        next_resolve = _to_utc(updates["resolve_time"]) if updates["resolve_time"] else None
    else:
        next_resolve = current_resolve

    _validate_times(next_open, next_close, next_resolve)

    previous_status = market.status
    changes: dict[str, dict[str, Any]] = {}

    if "title" in updates and updates["title"] != market.title:
        changes["title"] = {"from": market.title, "to": updates["title"]}
        market.title = updates["title"]

    if "description" in updates and updates["description"] != market.description:
        changes["description"] = {"from": market.description, "to": updates["description"]}
        market.description = updates["description"]

    if "status" in updates:
        next_status = TradingMarketStatus(updates["status"])
        if next_status != market.status:
            changes["status"] = {"from": _enum_value(market.status), "to": _enum_value(next_status)}
            market.status = next_status

    if next_open != current_open:
        changes["open_time"] = {"from": current_open.isoformat(), "to": next_open.isoformat()}
        market.open_time = next_open
        market.trading_starts_at = next_open

    if next_close != current_close:
        changes["close_time"] = {"from": current_close.isoformat(), "to": next_close.isoformat()}
        market.close_time = next_close
        market.trading_ends_at = next_close

    if next_resolve != current_resolve:
        changes["resolve_time"] = {
            "from": current_resolve.isoformat() if current_resolve else None,
            "to": next_resolve.isoformat() if next_resolve else None,
        }
        market.resolve_time = next_resolve

    market.resolves_after = next_resolve or next_close

    if "rules_json" in updates:
        rule = _get_active_market_rule(db, market.id)
        next_rules = updates["rules_json"] or {}
        if rule is None:
            rule = TradingMarketRule(
                market_id=market.id,
                version=1,
                rule_text=_rules_to_text(next_rules),
                rules_json=next_rules,
                is_active=True,
                created_by=actor_id,
                created_at=now,
                updated_at=now,
            )
            db.add(rule)
            changes["rules_json"] = {"from": None, "to": next_rules}
        elif rule.rules_json != next_rules:
            changes["rules_json"] = {"from": rule.rules_json, "to": next_rules}
            rule.rules_json = next_rules
            rule.rule_text = _rules_to_text(next_rules)
            rule.updated_at = now

    if not changes:
        return _serialize_market(market)

    market.updated_by = actor_id
    market.updated_at = now

    from_status = previous_status if previous_status != market.status else None
    to_status = market.status
    db.add(
        TradingMarketStateEvent(
            market_id=market.id,
            from_status=from_status,
            to_status=to_status,
            reason="admin_update",
            event_metadata={"event": "updated", "source": "admin_api_key", "changes": changes},
            changed_by=actor_id,
            created_at=now,
        )
    )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise _error_response_from_integrity(exc) from exc

    db.refresh(market)
    return _serialize_market(market)
