BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS trading;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'market_category' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.market_category AS ENUM ('politics', 'sports', 'finance', 'entertainment');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'market_status' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.market_status AS ENUM (
      'draft',
      'review',
      'scheduled',
      'live',
      'trading_paused',
      'closed',
      'resolving',
      'settled',
      'cancelled'
    );
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'market_outcome' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.market_outcome AS ENUM ('yes', 'no', 'void');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'order_side' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.order_side AS ENUM ('buy_yes', 'buy_no');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'order_type' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.order_type AS ENUM ('limit', 'market');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'order_tif' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.order_tif AS ENUM ('gtc', 'ioc', 'fok');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'order_status' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.order_status AS ENUM (
      'open',
      'partially_filled',
      'filled',
      'cancelled',
      'expired',
      'rejected'
    );
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'fill_settlement_state' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.fill_settlement_state AS ENUM ('pending', 'settled', 'reversed');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'wallet_account_type' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.wallet_account_type AS ENUM ('cash', 'bonus', 'locked_margin', 'fees', 'house');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'wallet_account_status' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.wallet_account_status AS ENUM ('active', 'locked', 'closed');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'ledger_direction' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.ledger_direction AS ENUM ('debit', 'credit');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'ledger_entry_type' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.ledger_entry_type AS ENUM (
      'deposit',
      'withdrawal',
      'order_reserve',
      'order_release',
      'trade_fill',
      'settlement',
      'fee',
      'adjustment',
      'refund'
    );
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'resolution_status' AND n.nspname = 'trading'
  ) THEN
    CREATE TYPE trading.resolution_status AS ENUM ('proposed', 'finalized', 'reversed');
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION trading.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS trading.markets (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  slug text NOT NULL,
  title text NOT NULL,
  description text,
  category trading.market_category NOT NULL,
  status trading.market_status NOT NULL DEFAULT 'draft',
  yes_price_bps integer NOT NULL DEFAULT 5000,
  no_price_bps integer GENERATED ALWAYS AS (10000 - yes_price_bps) STORED,
  tick_size_bps integer NOT NULL DEFAULT 1,
  min_order_size_minor bigint NOT NULL DEFAULT 100,
  max_order_size_minor bigint NOT NULL DEFAULT 500000000,
  trading_starts_at timestamptz,
  trading_ends_at timestamptz NOT NULL,
  resolves_after timestamptz NOT NULL,
  settled_outcome trading.market_outcome,
  settled_at timestamptz,
  created_by uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  updated_by uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT markets_public_id_uniq UNIQUE (public_id),
  CONSTRAINT markets_slug_uniq UNIQUE (slug),
  CONSTRAINT markets_yes_price_bps_chk CHECK (yes_price_bps BETWEEN 0 AND 10000),
  CONSTRAINT markets_tick_size_bps_chk CHECK (tick_size_bps BETWEEN 1 AND 500),
  CONSTRAINT markets_min_order_size_minor_chk CHECK (min_order_size_minor > 0),
  CONSTRAINT markets_max_order_size_minor_chk CHECK (max_order_size_minor >= min_order_size_minor),
  CONSTRAINT markets_schedule_chk CHECK (
    trading_starts_at IS NULL OR trading_starts_at <= trading_ends_at
  ),
  CONSTRAINT markets_resolution_time_chk CHECK (trading_ends_at <= resolves_after),
  CONSTRAINT markets_settlement_fields_chk CHECK (
    (status = 'settled' AND settled_outcome IS NOT NULL AND settled_at IS NOT NULL)
    OR (status <> 'settled' AND settled_outcome IS NULL AND settled_at IS NULL)
  )
);

DROP TRIGGER IF EXISTS trg_markets_set_updated_at ON trading.markets;
CREATE TRIGGER trg_markets_set_updated_at
BEFORE UPDATE ON trading.markets
FOR EACH ROW
EXECUTE FUNCTION trading.set_updated_at();

CREATE TABLE IF NOT EXISTS trading.market_rules (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  market_id bigint NOT NULL REFERENCES trading.markets(id) ON DELETE CASCADE,
  version integer NOT NULL,
  rule_text text NOT NULL,
  resolution_source_name text,
  resolution_source_url text,
  cutoff_at timestamptz,
  dispute_window_seconds integer NOT NULL DEFAULT 0,
  is_active boolean NOT NULL DEFAULT TRUE,
  created_by uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT market_rules_public_id_uniq UNIQUE (public_id),
  CONSTRAINT market_rules_market_version_uniq UNIQUE (market_id, version),
  CONSTRAINT market_rules_version_chk CHECK (version > 0),
  CONSTRAINT market_rules_dispute_window_chk CHECK (dispute_window_seconds >= 0)
);

CREATE TABLE IF NOT EXISTS trading.market_state_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  market_id bigint NOT NULL REFERENCES trading.markets(id) ON DELETE CASCADE,
  from_status trading.market_status,
  to_status trading.market_status NOT NULL,
  reason text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  changed_by uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT market_state_events_public_id_uniq UNIQUE (public_id),
  CONSTRAINT market_state_events_transition_chk CHECK (
    from_status IS NULL OR from_status <> to_status
  )
);

CREATE TABLE IF NOT EXISTS trading.orders (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  market_id bigint NOT NULL REFERENCES trading.markets(id) ON DELETE RESTRICT,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,
  side trading.order_side NOT NULL,
  order_type trading.order_type NOT NULL,
  time_in_force trading.order_tif NOT NULL DEFAULT 'gtc',
  status trading.order_status NOT NULL DEFAULT 'open',
  price_bps integer,
  quantity_contracts bigint NOT NULL,
  remaining_contracts bigint NOT NULL,
  reserved_cash_minor bigint NOT NULL DEFAULT 0,
  filled_cash_minor bigint NOT NULL DEFAULT 0,
  avg_fill_price_bps integer,
  idempotency_key text NOT NULL,
  client_order_id text,
  submitted_at timestamptz NOT NULL DEFAULT NOW(),
  expires_at timestamptz,
  cancelled_at timestamptz,
  cancel_reason text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT orders_public_id_uniq UNIQUE (public_id),
  CONSTRAINT orders_user_idempotency_uniq UNIQUE (user_id, idempotency_key),
  CONSTRAINT orders_user_client_order_uniq UNIQUE (user_id, client_order_id),
  CONSTRAINT orders_quantity_contracts_chk CHECK (quantity_contracts > 0),
  CONSTRAINT orders_remaining_contracts_chk CHECK (
    remaining_contracts >= 0 AND remaining_contracts <= quantity_contracts
  ),
  CONSTRAINT orders_reserved_cash_minor_chk CHECK (reserved_cash_minor >= 0),
  CONSTRAINT orders_filled_cash_minor_chk CHECK (filled_cash_minor >= 0),
  CONSTRAINT orders_avg_fill_price_bps_chk CHECK (
    avg_fill_price_bps IS NULL OR avg_fill_price_bps BETWEEN 1 AND 9999
  ),
  CONSTRAINT orders_price_model_chk CHECK (
    (order_type = 'limit' AND price_bps IS NOT NULL AND price_bps BETWEEN 1 AND 9999)
    OR (order_type = 'market' AND price_bps IS NULL)
  )
);

DROP TRIGGER IF EXISTS trg_orders_set_updated_at ON trading.orders;
CREATE TRIGGER trg_orders_set_updated_at
BEFORE UPDATE ON trading.orders
FOR EACH ROW
EXECUTE FUNCTION trading.set_updated_at();

CREATE TABLE IF NOT EXISTS trading.fills (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  market_id bigint NOT NULL REFERENCES trading.markets(id) ON DELETE RESTRICT,
  maker_order_id bigint NOT NULL REFERENCES trading.orders(id) ON DELETE RESTRICT,
  taker_order_id bigint NOT NULL REFERENCES trading.orders(id) ON DELETE RESTRICT,
  maker_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,
  taker_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,
  executed_side trading.order_side NOT NULL,
  price_bps integer NOT NULL,
  quantity_contracts bigint NOT NULL,
  notional_cash_minor bigint NOT NULL,
  fee_cash_minor bigint NOT NULL DEFAULT 0,
  settlement_state trading.fill_settlement_state NOT NULL DEFAULT 'pending',
  executed_at timestamptz NOT NULL DEFAULT NOW(),
  created_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT fills_public_id_uniq UNIQUE (public_id),
  CONSTRAINT fills_price_bps_chk CHECK (price_bps BETWEEN 1 AND 9999),
  CONSTRAINT fills_quantity_contracts_chk CHECK (quantity_contracts > 0),
  CONSTRAINT fills_notional_cash_minor_chk CHECK (notional_cash_minor >= 0),
  CONSTRAINT fills_fee_cash_minor_chk CHECK (fee_cash_minor >= 0),
  CONSTRAINT fills_orders_distinct_chk CHECK (maker_order_id <> taker_order_id),
  CONSTRAINT fills_users_distinct_chk CHECK (maker_user_id <> taker_user_id)
);

CREATE TABLE IF NOT EXISTS trading.positions (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  market_id bigint NOT NULL REFERENCES trading.markets(id) ON DELETE RESTRICT,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE RESTRICT,
  yes_contracts bigint NOT NULL DEFAULT 0,
  no_contracts bigint NOT NULL DEFAULT 0,
  avg_yes_entry_bps integer,
  avg_no_entry_bps integer,
  realized_pnl_minor bigint NOT NULL DEFAULT 0,
  unrealized_pnl_minor bigint NOT NULL DEFAULT 0,
  last_mark_bps integer NOT NULL DEFAULT 5000,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT positions_public_id_uniq UNIQUE (public_id),
  CONSTRAINT positions_market_user_uniq UNIQUE (market_id, user_id),
  CONSTRAINT positions_yes_contracts_chk CHECK (yes_contracts >= 0),
  CONSTRAINT positions_no_contracts_chk CHECK (no_contracts >= 0),
  CONSTRAINT positions_avg_yes_entry_bps_chk CHECK (
    avg_yes_entry_bps IS NULL OR avg_yes_entry_bps BETWEEN 1 AND 9999
  ),
  CONSTRAINT positions_avg_no_entry_bps_chk CHECK (
    avg_no_entry_bps IS NULL OR avg_no_entry_bps BETWEEN 1 AND 9999
  ),
  CONSTRAINT positions_last_mark_bps_chk CHECK (last_mark_bps BETWEEN 0 AND 10000)
);

DROP TRIGGER IF EXISTS trg_positions_set_updated_at ON trading.positions;
CREATE TRIGGER trg_positions_set_updated_at
BEFORE UPDATE ON trading.positions
FOR EACH ROW
EXECUTE FUNCTION trading.set_updated_at();

CREATE TABLE IF NOT EXISTS trading.wallet_accounts (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE RESTRICT,
  account_type trading.wallet_account_type NOT NULL,
  status trading.wallet_account_status NOT NULL DEFAULT 'active',
  currency_code char(3) NOT NULL DEFAULT 'DKK',
  balance_minor bigint NOT NULL DEFAULT 0,
  available_minor bigint NOT NULL DEFAULT 0,
  locked_minor bigint GENERATED ALWAYS AS (balance_minor - available_minor) STORED,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT wallet_accounts_public_id_uniq UNIQUE (public_id),
  CONSTRAINT wallet_accounts_user_type_currency_uniq UNIQUE (user_id, account_type, currency_code),
  CONSTRAINT wallet_accounts_currency_code_chk CHECK (currency_code = UPPER(currency_code)),
  CONSTRAINT wallet_accounts_balance_minor_chk CHECK (balance_minor >= 0),
  CONSTRAINT wallet_accounts_available_minor_chk CHECK (
    available_minor >= 0 AND available_minor <= balance_minor
  )
);

DROP TRIGGER IF EXISTS trg_wallet_accounts_set_updated_at ON trading.wallet_accounts;
CREATE TRIGGER trg_wallet_accounts_set_updated_at
BEFORE UPDATE ON trading.wallet_accounts
FOR EACH ROW
EXECUTE FUNCTION trading.set_updated_at();

CREATE TABLE IF NOT EXISTS trading.ledger_entries (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  transaction_id uuid NOT NULL,
  leg_index smallint NOT NULL,
  account_id bigint NOT NULL REFERENCES trading.wallet_accounts(id) ON DELETE RESTRICT,
  direction trading.ledger_direction NOT NULL,
  amount_minor bigint NOT NULL,
  currency_code char(3) NOT NULL DEFAULT 'DKK',
  entry_type trading.ledger_entry_type NOT NULL,
  market_id bigint REFERENCES trading.markets(id) ON DELETE SET NULL,
  order_id bigint REFERENCES trading.orders(id) ON DELETE SET NULL,
  fill_id bigint REFERENCES trading.fills(id) ON DELETE SET NULL,
  user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  note text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT ledger_entries_public_id_uniq UNIQUE (public_id),
  CONSTRAINT ledger_entries_tx_leg_uniq UNIQUE (transaction_id, leg_index),
  CONSTRAINT ledger_entries_leg_index_chk CHECK (leg_index > 0),
  CONSTRAINT ledger_entries_amount_minor_chk CHECK (amount_minor > 0),
  CONSTRAINT ledger_entries_currency_code_chk CHECK (currency_code = UPPER(currency_code))
);

CREATE OR REPLACE FUNCTION trading.assert_balanced_ledger_transaction()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  tx_id uuid;
  signed_total bigint;
  leg_count integer;
BEGIN
  tx_id := NEW.transaction_id;

  SELECT
    COALESCE(
      SUM(
        CASE direction
          WHEN 'debit' THEN amount_minor
          ELSE -amount_minor
        END
      ),
      0
    ),
    COUNT(*)
  INTO signed_total, leg_count
  FROM trading.ledger_entries
  WHERE transaction_id = tx_id;

  IF leg_count < 2 THEN
    RAISE EXCEPTION 'Ledger transaction % must contain at least 2 legs', tx_id;
  END IF;

  IF signed_total <> 0 THEN
    RAISE EXCEPTION 'Ledger transaction % is not balanced (difference=%)', tx_id, signed_total;
  END IF;

  RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_assert_balanced_ledger_transaction ON trading.ledger_entries;
CREATE CONSTRAINT TRIGGER trg_assert_balanced_ledger_transaction
AFTER INSERT OR UPDATE ON trading.ledger_entries
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION trading.assert_balanced_ledger_transaction();

CREATE TABLE IF NOT EXISTS trading.resolutions (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  market_id bigint NOT NULL REFERENCES trading.markets(id) ON DELETE RESTRICT,
  version integer NOT NULL DEFAULT 1,
  status trading.resolution_status NOT NULL DEFAULT 'proposed',
  outcome trading.market_outcome NOT NULL,
  resolved_by uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  resolution_source_name text,
  resolution_source_url text,
  notes text,
  dispute_deadline_at timestamptz,
  resolved_at timestamptz NOT NULL DEFAULT NOW(),
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT resolutions_public_id_uniq UNIQUE (public_id),
  CONSTRAINT resolutions_market_version_uniq UNIQUE (market_id, version),
  CONSTRAINT resolutions_version_chk CHECK (version > 0)
);

DROP TRIGGER IF EXISTS trg_resolutions_set_updated_at ON trading.resolutions;
CREATE TRIGGER trg_resolutions_set_updated_at
BEFORE UPDATE ON trading.resolutions
FOR EACH ROW
EXECUTE FUNCTION trading.set_updated_at();

CREATE TABLE IF NOT EXISTS trading.audit_logs (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  public_id uuid NOT NULL DEFAULT gen_random_uuid(),
  actor_user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  actor_role text,
  action text NOT NULL,
  resource_type text NOT NULL,
  resource_id bigint,
  resource_public_id uuid,
  request_id text,
  ip_address inet,
  user_agent text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  CONSTRAINT audit_logs_public_id_uniq UNIQUE (public_id)
);

CREATE INDEX IF NOT EXISTS idx_markets_status_ends_at
  ON trading.markets(status, trading_ends_at);

CREATE INDEX IF NOT EXISTS idx_markets_category_status
  ON trading.markets(category, status);

CREATE INDEX IF NOT EXISTS idx_market_state_events_market_created_at
  ON trading.market_state_events(market_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_market_status_price_submitted
  ON trading.orders(market_id, status, price_bps, submitted_at);

CREATE INDEX IF NOT EXISTS idx_orders_user_submitted_at
  ON trading.orders(user_id, submitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_fills_market_executed_at
  ON trading.fills(market_id, executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_fills_maker_user_executed_at
  ON trading.fills(maker_user_id, executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_fills_taker_user_executed_at
  ON trading.fills(taker_user_id, executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_positions_user_market
  ON trading.positions(user_id, market_id);

CREATE INDEX IF NOT EXISTS idx_wallet_accounts_user_type
  ON trading.wallet_accounts(user_id, account_type);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_account_created_at
  ON trading.ledger_entries(account_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_transaction_id
  ON trading.ledger_entries(transaction_id);

CREATE INDEX IF NOT EXISTS idx_resolutions_market_status
  ON trading.resolutions(market_id, status);

CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_created_at
  ON trading.audit_logs(actor_user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_created_at
  ON trading.audit_logs(resource_type, resource_id, created_at DESC);

REVOKE ALL ON ALL TABLES IN SCHEMA trading FROM anon, authenticated;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA trading FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA trading REVOKE ALL ON TABLES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA trading REVOKE ALL ON SEQUENCES FROM anon, authenticated;

COMMIT;
