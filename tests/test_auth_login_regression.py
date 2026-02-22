import importlib
import sys

import pytest
from fastapi import HTTPException

pytest.importorskip("psycopg")


def _load_main_with_env(monkeypatch: pytest.MonkeyPatch, *, auth_secret: str | None):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/foresee")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_JWT_ISSUER", raising=False)

    if auth_secret is None:
        monkeypatch.delenv("AUTH_SECRET", raising=False)
    else:
        monkeypatch.setenv("AUTH_SECRET", auth_secret)

    sys.modules.pop("app.main", None)
    sys.modules.pop("app.storage", None)

    import app.main as main

    return importlib.reload(main)


def test_login_returns_503_when_auth_secret_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    main = _load_main_with_env(monkeypatch, auth_secret=None)

    with pytest.raises(HTTPException) as exc_info:
        main.login(main.LoginRequest(email="ops@example.com", password="password123"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Legacy auth endpoints are disabled."


def test_login_returns_503_if_signing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    main = _load_main_with_env(monkeypatch, auth_secret=None)
    monkeypatch.setattr(main, "_require_legacy_auth_secret", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        main.login(main.LoginRequest(email="ops@example.com", password="password123"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Legacy auth endpoints are disabled."
