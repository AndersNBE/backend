BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE trading.markets
  ADD COLUMN IF NOT EXISTS open_time timestamptz,
  ADD COLUMN IF NOT EXISTS close_time timestamptz,
  ADD COLUMN IF NOT EXISTS resolve_time timestamptz;

UPDATE trading.markets
SET
  open_time = COALESCE(open_time, trading_starts_at, created_at, NOW()),
  close_time = COALESCE(close_time, trading_ends_at, NOW() + INTERVAL '30 days'),
  resolve_time = COALESCE(resolve_time, resolves_after);

UPDATE trading.markets
SET
  trading_starts_at = COALESCE(trading_starts_at, open_time),
  trading_ends_at = COALESCE(trading_ends_at, close_time),
  resolves_after = COALESCE(resolves_after, resolve_time, close_time);

ALTER TABLE trading.markets
  ALTER COLUMN open_time SET NOT NULL,
  ALTER COLUMN close_time SET NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'markets_open_before_close_chk'
      AND connamespace = 'trading'::regnamespace
  ) THEN
    ALTER TABLE trading.markets
      ADD CONSTRAINT markets_open_before_close_chk
      CHECK (open_time < close_time) NOT VALID;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'markets_resolve_after_close_chk'
      AND connamespace = 'trading'::regnamespace
  ) THEN
    ALTER TABLE trading.markets
      ADD CONSTRAINT markets_resolve_after_close_chk
      CHECK (resolve_time IS NULL OR resolve_time >= close_time) NOT VALID;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'markets_slug_format_chk'
      AND connamespace = 'trading'::regnamespace
  ) THEN
    ALTER TABLE trading.markets
      ADD CONSTRAINT markets_slug_format_chk
      CHECK (slug ~ '^[a-z0-9_]+$') NOT VALID;
  END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_markets_category_status
  ON trading.markets(category, status);

CREATE INDEX IF NOT EXISTS idx_markets_close_time
  ON trading.markets(close_time);

CREATE INDEX IF NOT EXISTS idx_market_state_events_market_created_at
  ON trading.market_state_events(market_id, created_at DESC);

ALTER TABLE trading.market_rules
  ADD COLUMN IF NOT EXISTS rules_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

UPDATE trading.market_rules
SET rules_json = jsonb_strip_nulls(
  jsonb_build_object(
    'rule_text', rule_text,
    'resolution_source_name', resolution_source_name,
    'resolution_source_url', resolution_source_url,
    'cutoff_at', cutoff_at,
    'dispute_window_seconds', dispute_window_seconds
  )
)
WHERE rules_json = '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS idx_market_rules_market_active_unique
  ON trading.market_rules(market_id)
  WHERE is_active;

DROP TRIGGER IF EXISTS trg_market_rules_set_updated_at ON trading.market_rules;
CREATE TRIGGER trg_market_rules_set_updated_at
BEFORE UPDATE ON trading.market_rules
FOR EACH ROW
EXECUTE FUNCTION trading.set_updated_at();

DO $$
DECLARE
  v_table_name text;
BEGIN
  FOR v_table_name IN
    SELECT tablename
    FROM pg_tables
    WHERE schemaname = 'trading'
  LOOP
    EXECUTE format('ALTER TABLE trading.%I ENABLE ROW LEVEL SECURITY', v_table_name);
  END LOOP;
END;
$$;

DO $$
DECLARE
  v_table_name text;
  policy_name text;
BEGIN
  FOR v_table_name IN
    SELECT tablename
    FROM pg_tables
    WHERE schemaname = 'trading'
  LOOP
    policy_name := v_table_name || '_service_role_all';
    IF NOT EXISTS (
      SELECT 1
      FROM pg_policies
      WHERE schemaname = 'trading'
        AND tablename = v_table_name
        AND policyname = policy_name
    ) THEN
      EXECUTE format(
        'CREATE POLICY %I ON trading.%I FOR ALL TO service_role USING (TRUE) WITH CHECK (TRUE)',
        policy_name,
        v_table_name
      );
    END IF;
  END LOOP;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'trading'
      AND tablename = 'markets'
      AND policyname = 'markets_public_visible_read'
  ) THEN
    CREATE POLICY markets_public_visible_read
      ON trading.markets
      FOR SELECT
      TO anon, authenticated
      USING (status IN ('live', 'trading_paused', 'closed', 'resolving', 'settled', 'cancelled'));
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'trading'
      AND tablename = 'market_rules'
      AND policyname = 'market_rules_public_visible_read'
  ) THEN
    CREATE POLICY market_rules_public_visible_read
      ON trading.market_rules
      FOR SELECT
      TO anon, authenticated
      USING (
        EXISTS (
          SELECT 1
          FROM trading.markets m
          WHERE m.id = market_rules.market_id
            AND m.status IN ('live', 'trading_paused', 'closed', 'resolving', 'settled', 'cancelled')
        )
      );
  END IF;

END;
$$;

GRANT SELECT ON trading.markets TO anon, authenticated;
GRANT SELECT ON trading.market_rules TO anon, authenticated;

REVOKE INSERT, UPDATE, DELETE ON trading.markets FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE ON trading.market_rules FROM anon, authenticated;

CREATE OR REPLACE FUNCTION trading.audit_market_write()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  v_action text;
BEGIN
  v_action := CASE WHEN TG_OP = 'INSERT' THEN 'market_created' ELSE 'market_updated' END;

  INSERT INTO trading.audit_logs (
    actor_user_id,
    actor_role,
    action,
    resource_type,
    resource_id,
    resource_public_id,
    metadata
  )
  VALUES (
    COALESCE(NEW.updated_by, NEW.created_by),
    'admin_api',
    v_action,
    'market',
    NEW.id,
    NEW.public_id,
    jsonb_build_object(
      'slug', NEW.slug,
      'status', NEW.status,
      'open_time', NEW.open_time,
      'close_time', NEW.close_time,
      'resolve_time', NEW.resolve_time
    )
  );

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_audit_markets_write ON trading.markets;
CREATE TRIGGER trg_audit_markets_write
AFTER INSERT OR UPDATE ON trading.markets
FOR EACH ROW
EXECUTE FUNCTION trading.audit_market_write();

COMMIT;
