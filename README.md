# Backend Runtime Notes (Railway)

## Required environment variables

- `ENVIRONMENT=production`
- `DATABASE_URL=<Supabase Postgres connection string>`
- `SUPABASE_URL=https://<project-ref>.supabase.co`
- `SUPABASE_JWT_ISSUER=https://<project-ref>.supabase.co/auth/v1`
- `SUPABASE_JWT_AUDIENCE=authenticated`
- `REDIS_URL=redis://...` or `rediss://...` (Railway Redis endpoint)
- `ADMIN_API_KEY=<long random value>` (required for admin market create/update endpoints)

`DATABASE_URL` must be PostgreSQL. Startup now fails if it is missing.
SSL is enforced by appending `sslmode=require` if it is not already present.

Optional Redis/rate limiter tuning:

- `REDIS_INIT_TIMEOUT_SECONDS=2.5`
- `RATE_LIMITER_RETRY_LOOP_SECONDS=10`
- `RATE_LIMITER_STARTUP_WAIT_SECONDS=0.25`
- `RATE_LIMITER_FAIL_CLOSED=false` (set `true` to fail startup if Redis is unavailable)
- `REDIS_SSL_CERT_REQS=required` (`none`, `optional`, `required`)
- `REDIS_FORCE_TLS=false` (set `true` to force `redis://` URLs to connect as TLS)
- `REDIS_TRY_TLS_FALLBACK=true` (tries `rediss://` automatically if `redis://` fails)

## Trading API source of truth

- Markets now come from `trading.markets` in Postgres.
- New router:
  - `GET /trading/markets`
  - `GET /trading/markets/{slug}`
  - `POST /trading/admin/markets` (admin key required)
  - `PATCH /trading/admin/markets/{market_id}` (admin key required)
- Legacy `GET /markets` now reads from the same trading table (frontend-compatible shape).

Frontend should continue using Supabase only for auth and call FastAPI for trading data.

Public endpoints only return visible market statuses (`live`, `trading_paused`, `closed`, `resolving`, `settled`, `cancelled`).
`draft`, `review`, and `scheduled` are hidden from public reads.

## Migrations

Apply trading migrations in order:

1. `backend/supabase/migrations/20260219_001_trading_core.sql`
2. `backend/supabase/migrations/20260222_002_trading_admin_visibility.sql`

If you use Supabase SQL editor, run each file as a full script in sequence.
If you use CLI tooling, execute both scripts against the target database before deploying backend changes.

## Smoke test plan

Replace `${API}` with the deployed backend URL:

```bash
curl -i "${API}/health"
```

Expect: `200` and `{"status":"ok"}`.

```bash
curl -i "${API}/markets"
```

Expect: `200` and JSON array of market objects.

```bash
curl -i -X POST "${API}/trading/admin/markets" \
  -H "X-Admin-Key: ${ADMIN_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "slug":"demo_market_1",
    "title":"Will Udfall launch new mobile app this quarter?",
    "category":"finance",
    "open_time":"2026-03-01T12:00:00Z",
    "close_time":"2026-03-31T12:00:00Z",
    "resolve_time":"2026-04-01T12:00:00Z",
    "rules_json":{"rule":"Resolves YES if app is publicly released on iOS or Android before quarter end."}
  }'
```

Expect: `201` and created market with `status: "draft"`.

```bash
curl -i -X PATCH "${API}/trading/admin/markets/<market_id>" \
  -H "X-Admin-Key: ${ADMIN_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"status":"live"}'
```

Expect: `200` and updated market with status `live`.

## Seed script

Development seed:

```bash
python -m app.scripts.seed_trading_markets
```

By default, seeding is blocked in production. To override intentionally:

```bash
ALLOW_PRODUCTION_SEED=true python -m app.scripts.seed_trading_markets
```

## CI runtime check

```bash
pytest -q tests/test_db_runtime.py
```

Requires `DATABASE_URL` in CI. The test creates an engine from env and runs `SELECT 1`.

Additional tests:

```bash
pytest -q tests/test_trading_validation.py
pytest -q tests/test_trading_admin_integration.py
```

`tests/test_trading_admin_integration.py` is skipped unless `DATABASE_URL` is available.
