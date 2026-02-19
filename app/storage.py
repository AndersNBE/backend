import enum
import os
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Computed,
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL_RAW = os.getenv("DATABASE_URL")


def _normalize_database_url(raw_url: str) -> str:
    database_url = raw_url.strip()
    if database_url.startswith("postgres://"):
        database_url = f"postgresql://{database_url[len('postgres://'):]}"

    parsed = make_url(database_url)

    if parsed.drivername.startswith("postgresql") or parsed.drivername.startswith("postgres"):
        parsed = parsed.set(drivername="postgresql+psycopg")
    else:
        raise RuntimeError(
            f"Unsupported DATABASE_URL driver '{parsed.drivername}'. Use a PostgreSQL URL."
        )

    query = dict(parsed.query)
    query.setdefault("sslmode", "require")
    parsed = parsed.set(query=query)
    return parsed.render_as_string(hide_password=False)


if not DATABASE_URL_RAW:
    raise RuntimeError(
        "Missing DATABASE_URL. SQLite fallback has been removed. "
        "Set DATABASE_URL to the Supabase Postgres connection string."
    )

DATABASE_URL = _normalize_database_url(DATABASE_URL_RAW)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT_SECONDS", "30")),
    future=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
TradingBase = declarative_base()


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (UniqueConstraint("user_id", "key", "endpoint", name="uq_idempotency"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), index=True, nullable=False)
    key = Column(String(128), index=True, nullable=False)
    endpoint = Column(String(256), nullable=False)
    request_hash = Column(String(64), nullable=False)
    response_body = Column(Text, nullable=True)
    status_code = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)


class BetRecord(Base):
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), index=True, nullable=False)
    market_id = Column(String(128), nullable=False)
    outcome = Column(String(16), nullable=False)
    stake_dkk = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TradingMarketCategory(str, enum.Enum):
    politics = "politics"
    sports = "sports"
    finance = "finance"
    entertainment = "entertainment"


class TradingMarketStatus(str, enum.Enum):
    draft = "draft"
    review = "review"
    scheduled = "scheduled"
    live = "live"
    trading_paused = "trading_paused"
    closed = "closed"
    resolving = "resolving"
    settled = "settled"
    cancelled = "cancelled"


class TradingOrderSide(str, enum.Enum):
    buy_yes = "buy_yes"
    buy_no = "buy_no"


class TradingOrderType(str, enum.Enum):
    limit = "limit"
    market = "market"


class TradingOrderTif(str, enum.Enum):
    gtc = "gtc"
    ioc = "ioc"
    fok = "fok"


class TradingOrderStatus(str, enum.Enum):
    open = "open"
    partially_filled = "partially_filled"
    filled = "filled"
    cancelled = "cancelled"
    expired = "expired"
    rejected = "rejected"


class TradingMarket(TradingBase):
    __tablename__ = "markets"
    __table_args__ = {"schema": "trading"}

    id = Column(BigInteger, primary_key=True)
    public_id = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    slug = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(
        Enum(
            TradingMarketCategory,
            name="market_category",
            schema="trading",
            native_enum=True,
            create_type=False,
        ),
        nullable=False,
    )
    status = Column(
        Enum(
            TradingMarketStatus,
            name="market_status",
            schema="trading",
            native_enum=True,
            create_type=False,
        ),
        nullable=False,
        default=TradingMarketStatus.draft,
    )
    yes_price_bps = Column(Integer, nullable=False, default=5000, server_default=text("5000"))
    no_price_bps = Column(Integer, Computed("10000 - yes_price_bps"), nullable=False)
    tick_size_bps = Column(Integer, nullable=False, default=1, server_default=text("1"))
    min_order_size_minor = Column(BigInteger, nullable=False, default=100, server_default=text("100"))
    max_order_size_minor = Column(
        BigInteger,
        nullable=False,
        default=500000000,
        server_default=text("500000000"),
    )
    trading_starts_at = Column(DateTime(timezone=True), nullable=True)
    trading_ends_at = Column(DateTime(timezone=True), nullable=False)
    resolves_after = Column(DateTime(timezone=True), nullable=False)
    settled_outcome = Column(String(16), nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class TradingMarketRule(TradingBase):
    __tablename__ = "market_rules"
    __table_args__ = {"schema": "trading"}

    id = Column(BigInteger, primary_key=True)
    public_id = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    market_id = Column(BigInteger, nullable=False)
    version = Column(Integer, nullable=False)
    rule_text = Column(Text, nullable=False)
    resolution_source_name = Column(Text, nullable=True)
    resolution_source_url = Column(Text, nullable=True)
    cutoff_at = Column(DateTime(timezone=True), nullable=True)
    dispute_window_seconds = Column(Integer, nullable=False, default=0, server_default=text("0"))
    is_active = Column(Boolean, nullable=False, default=True, server_default=text("true"))
    created_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class TradingOrder(TradingBase):
    __tablename__ = "orders"
    __table_args__ = {"schema": "trading"}

    id = Column(BigInteger, primary_key=True)
    public_id = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    market_id = Column(BigInteger, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    side = Column(
        Enum(
            TradingOrderSide,
            name="order_side",
            schema="trading",
            native_enum=True,
            create_type=False,
        ),
        nullable=False,
    )
    order_type = Column(
        Enum(
            TradingOrderType,
            name="order_type",
            schema="trading",
            native_enum=True,
            create_type=False,
        ),
        nullable=False,
    )
    time_in_force = Column(
        Enum(
            TradingOrderTif,
            name="order_tif",
            schema="trading",
            native_enum=True,
            create_type=False,
        ),
        nullable=False,
        default=TradingOrderTif.gtc,
    )
    status = Column(
        Enum(
            TradingOrderStatus,
            name="order_status",
            schema="trading",
            native_enum=True,
            create_type=False,
        ),
        nullable=False,
        default=TradingOrderStatus.open,
    )
    price_bps = Column(Integer, nullable=True)
    quantity_contracts = Column(BigInteger, nullable=False)
    remaining_contracts = Column(BigInteger, nullable=False)
    reserved_cash_minor = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    filled_cash_minor = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    avg_fill_price_bps = Column(Integer, nullable=True)
    idempotency_key = Column(Text, nullable=False)
    client_order_id = Column(Text, nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    cancel_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
