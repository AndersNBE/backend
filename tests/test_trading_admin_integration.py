import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

if not os.getenv("DATABASE_URL"):
    pytest.skip("DATABASE_URL is not set", allow_module_level=True)

from app.main import app
from app.storage import SessionLocal, TradingMarket, TradingMarketRule, TradingMarketStateEvent

ADMIN_KEY = "integration-admin-key"


@pytest.fixture(scope="module", autouse=True)
def _ensure_database_access():
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"Database is not reachable for integration tests: {exc}", allow_module_level=True)
    finally:
        db.close()


def _build_payload(slug: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "slug": slug,
        "title": f"Integration market {slug}",
        "description": "Integration test market",
        "category": "finance",
        "open_time": now.isoformat(),
        "close_time": (now + timedelta(days=3)).isoformat(),
        "resolve_time": (now + timedelta(days=4)).isoformat(),
        "rules_json": {"rule": "integration test"},
    }


def _cleanup_market(slug: str) -> None:
    db = SessionLocal()
    try:
        market = db.query(TradingMarket).filter(TradingMarket.slug == slug).one_or_none()
        if market is None:
            return
        db.query(TradingMarketStateEvent).filter(
            TradingMarketStateEvent.market_id == market.id
        ).delete(synchronize_session=False)
        db.query(TradingMarketRule).filter(
            TradingMarketRule.market_id == market.id
        ).delete(synchronize_session=False)
        db.delete(market)
        db.commit()
    finally:
        db.close()


def test_create_market_requires_admin_key() -> None:
    os.environ["ADMIN_API_KEY"] = ADMIN_KEY
    slug = f"it_admin_{uuid.uuid4().hex[:10]}"
    payload = _build_payload(slug)

    with TestClient(app) as client:
        no_key_response = client.post("/trading/admin/markets", json=payload)
        assert no_key_response.status_code == 401

        ok_response = client.post(
            "/trading/admin/markets",
            json=payload,
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert ok_response.status_code == 201
        body = ok_response.json()
        assert body["slug"] == slug
        assert body["status"] == "draft"
        assert body["id"]

    _cleanup_market(slug)


def test_public_list_only_returns_visible_markets() -> None:
    os.environ["ADMIN_API_KEY"] = ADMIN_KEY
    live_slug = f"it_live_{uuid.uuid4().hex[:10]}"
    draft_slug = f"it_draft_{uuid.uuid4().hex[:10]}"
    live_payload = _build_payload(live_slug)
    draft_payload = _build_payload(draft_slug)

    with TestClient(app) as client:
        live_create = client.post(
            "/trading/admin/markets",
            json=live_payload,
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert live_create.status_code == 201
        live_market_id = live_create.json()["id"]

        draft_create = client.post(
            "/trading/admin/markets",
            json=draft_payload,
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert draft_create.status_code == 201

        patch_live = client.patch(
            f"/trading/admin/markets/{live_market_id}",
            json={"status": "live"},
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert patch_live.status_code == 200

        public_list = client.get("/trading/markets")
        assert public_list.status_code == 200
        slugs = {item["slug"] for item in public_list.json()}
        assert live_slug in slugs
        assert draft_slug not in slugs

        hidden_market = client.get(f"/trading/markets/{draft_slug}")
        assert hidden_market.status_code == 404

    _cleanup_market(live_slug)
    _cleanup_market(draft_slug)
