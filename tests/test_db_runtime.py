import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


def _normalize_database_url(raw_url: str) -> str:
    database_url = raw_url.strip()
    if database_url.startswith("postgres://"):
        database_url = f"postgresql://{database_url[len('postgres://'):]}"

    parsed = make_url(database_url)
    if parsed.drivername.startswith("postgresql") or parsed.drivername.startswith("postgres"):
        parsed = parsed.set(drivername="postgresql+psycopg")
    else:
        raise RuntimeError(f"Unsupported DATABASE_URL driver '{parsed.drivername}'")

    query = dict(parsed.query)
    query.setdefault("sslmode", "require")
    parsed = parsed.set(query=query)
    return parsed.render_as_string(hide_password=False)


def test_database_runtime_smoke() -> None:
    raw_database_url = os.getenv("DATABASE_URL")
    if not raw_database_url:
        pytest.skip("DATABASE_URL is not set")

    engine = create_engine(
        _normalize_database_url(raw_database_url),
        pool_pre_ping=True,
        future=True,
    )
    try:
        with engine.connect() as connection:
            value = connection.execute(text("SELECT 1")).scalar_one()
            assert value == 1
    finally:
        engine.dispose()
