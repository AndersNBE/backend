from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.trading_validation import CreateAdminMarketRequest


def _valid_payload() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "slug": "finance_btc_2028",
        "title": "Will BTC close above 150000 in 2028?",
        "description": "Test market",
        "category": "finance",
        "open_time": now.isoformat(),
        "close_time": (now + timedelta(days=7)).isoformat(),
        "resolve_time": (now + timedelta(days=8)).isoformat(),
        "rules_json": {"rule": "official close"},
    }


def test_slug_validation_rejects_invalid_slug() -> None:
    payload = _valid_payload()
    payload["slug"] = "bad-slug"
    with pytest.raises(ValidationError):
        CreateAdminMarketRequest(**payload)


def test_time_validation_rejects_open_after_close() -> None:
    payload = _valid_payload()
    now = datetime.now(timezone.utc)
    payload["open_time"] = (now + timedelta(days=2)).isoformat()
    payload["close_time"] = (now + timedelta(days=1)).isoformat()
    with pytest.raises(ValidationError):
        CreateAdminMarketRequest(**payload)


def test_time_validation_rejects_resolve_before_close() -> None:
    payload = _valid_payload()
    now = datetime.now(timezone.utc)
    payload["open_time"] = now.isoformat()
    payload["close_time"] = (now + timedelta(days=2)).isoformat()
    payload["resolve_time"] = (now + timedelta(days=1)).isoformat()
    with pytest.raises(ValidationError):
        CreateAdminMarketRequest(**payload)
