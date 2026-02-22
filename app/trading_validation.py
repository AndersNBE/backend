from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TradingCategory = Literal["politics", "sports", "finance", "entertainment"]
TradingStatus = Literal[
    "draft",
    "review",
    "scheduled",
    "live",
    "trading_paused",
    "closed",
    "resolving",
    "settled",
    "cancelled",
]


def validate_market_times(open_time: datetime, close_time: datetime, resolve_time: datetime | None) -> None:
    if open_time >= close_time:
        raise ValueError("open_time must be earlier than close_time")
    if resolve_time is not None and resolve_time < close_time:
        raise ValueError("resolve_time must be greater than or equal to close_time")


class CreateAdminMarketRequest(BaseModel):
    slug: str = Field(..., min_length=3, max_length=160, pattern=r"^[a-z0-9_]+$")
    title: str = Field(..., min_length=3, max_length=240)
    description: str | None = Field(default=None, max_length=4000)
    category: TradingCategory
    open_time: datetime
    close_time: datetime
    resolve_time: datetime | None = None
    rules_json: dict[str, Any] = Field(default_factory=dict)
    yes_price_bps: int = Field(default=5000, ge=0, le=10000)
    tick_size_bps: int = Field(default=1, ge=1, le=500)
    min_order_size_minor: int = Field(default=100, ge=1)
    max_order_size_minor: int = Field(default=500000000, ge=1)

    @model_validator(mode="after")
    def validate_time_and_size_constraints(self):
        validate_market_times(self.open_time, self.close_time, self.resolve_time)
        if self.max_order_size_minor < self.min_order_size_minor:
            raise ValueError("max_order_size_minor must be greater than or equal to min_order_size_minor")
        return self


class UpdateAdminMarketRequest(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=240)
    description: str | None = Field(default=None, max_length=4000)
    status: TradingStatus | None = None
    open_time: datetime | None = None
    close_time: datetime | None = None
    resolve_time: datetime | None = None
    rules_json: dict[str, Any] | None = None
