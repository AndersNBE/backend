import enum
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..storage import (
    TradingMarket,
    TradingMarketCategory,
    TradingMarketRule,
    TradingMarketStatus,
    get_db,
)

router = APIRouter(prefix="/trading", tags=["trading"])

TradingCategory = Literal["politics", "sports", "finance", "entertainment"]


class TradingMarketResponse(BaseModel):
    id: str
    publicId: str
    slug: str
    title: str
    description: str | None = None
    category: TradingCategory
    status: str
    yesPriceBps: int
    noPriceBps: int
    yesPrice: float
    noPrice: float
    tickSizeBps: int
    minOrderSizeMinor: int
    maxOrderSizeMinor: int
    tradingEndsAt: datetime
    resolvesAfter: datetime
    createdAt: datetime
    updatedAt: datetime


class CreateTradingMarketRequest(BaseModel):
    slug: str = Field(
        ...,
        min_length=3,
        max_length=160,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    title: str = Field(..., min_length=3, max_length=240)
    description: str | None = Field(default=None, max_length=4000)
    category: TradingCategory
    tradingEndsAt: datetime | None = None
    resolvesAfter: datetime | None = None
    yesPriceBps: int = Field(default=5000, ge=0, le=10000)
    tickSizeBps: int = Field(default=1, ge=1, le=500)
    minOrderSizeMinor: int = Field(default=100, ge=1)
    maxOrderSizeMinor: int = Field(default=500000000, ge=1)
    ruleText: str | None = Field(default=None, min_length=1, max_length=5000)


def _enum_value(value: object) -> str:
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _serialize_market(market: TradingMarket) -> TradingMarketResponse:
    yes_price_bps = int(market.yes_price_bps)
    no_price_bps = int(getattr(market, "no_price_bps", 10000 - yes_price_bps))
    category = _enum_value(market.category)

    return TradingMarketResponse(
        id=str(market.id),
        publicId=str(market.public_id),
        slug=market.slug,
        title=market.title,
        description=market.description,
        category=category,  # type: ignore[arg-type]
        status=_enum_value(market.status),
        yesPriceBps=yes_price_bps,
        noPriceBps=no_price_bps,
        yesPrice=round(yes_price_bps / 10000, 5),
        noPrice=round(no_price_bps / 10000, 5),
        tickSizeBps=int(market.tick_size_bps),
        minOrderSizeMinor=int(market.min_order_size_minor),
        maxOrderSizeMinor=int(market.max_order_size_minor),
        tradingEndsAt=market.trading_ends_at,
        resolvesAfter=market.resolves_after,
        createdAt=market.created_at,
        updatedAt=market.updated_at,
    )


def _parse_optional_uuid(raw_value: str | None) -> uuid.UUID | None:
    if not raw_value:
        return None
    try:
        return uuid.UUID(raw_value)
    except (ValueError, TypeError):
        return None


@router.get("/markets", response_model=list[TradingMarketResponse])
def list_trading_markets(db: Session = Depends(get_db)):
    markets = (
        db.query(TradingMarket)
        .order_by(TradingMarket.created_at.desc())
        .all()
    )
    return [_serialize_market(market) for market in markets]


@router.get("/markets/{slug}", response_model=TradingMarketResponse)
def get_trading_market_by_slug(slug: str, db: Session = Depends(get_db)):
    market = (
        db.query(TradingMarket)
        .filter(TradingMarket.slug == slug)
        .one_or_none()
    )
    if market is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Market not found")
    return _serialize_market(market)


@router.post(
    "/markets",
    response_model=TradingMarketResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_trading_market(
    payload: CreateTradingMarketRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    if payload.maxOrderSizeMinor < payload.minOrderSizeMinor:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="maxOrderSizeMinor must be greater than or equal to minOrderSizeMinor",
        )

    now = datetime.now(timezone.utc)
    trading_ends_at = payload.tradingEndsAt or (now + timedelta(days=30))
    resolves_after = payload.resolvesAfter or (trading_ends_at + timedelta(days=1))

    if resolves_after < trading_ends_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="resolvesAfter must be greater than or equal to tradingEndsAt",
        )

    request_user_id = getattr(request.state, "user_id", None)
    actor_id = _parse_optional_uuid(str(request_user_id) if request_user_id else None)

    market = TradingMarket(
        slug=payload.slug,
        title=payload.title,
        description=payload.description,
        category=TradingMarketCategory(payload.category),
        status=TradingMarketStatus.draft,
        yes_price_bps=payload.yesPriceBps,
        tick_size_bps=payload.tickSizeBps,
        min_order_size_minor=payload.minOrderSizeMinor,
        max_order_size_minor=payload.maxOrderSizeMinor,
        trading_starts_at=None,
        trading_ends_at=trading_ends_at,
        resolves_after=resolves_after,
        created_by=actor_id,
        updated_by=actor_id,
        created_at=now,
        updated_at=now,
    )

    db.add(market)
    try:
        db.flush()
        if payload.ruleText:
            db.add(
                TradingMarketRule(
                    market_id=market.id,
                    version=1,
                    rule_text=payload.ruleText,
                    is_active=True,
                    created_by=actor_id,
                    created_at=now,
                )
            )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if "markets_slug_uniq" in str(exc.orig):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Market slug already exists",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create market",
        ) from exc

    db.refresh(market)
    return _serialize_market(market)
