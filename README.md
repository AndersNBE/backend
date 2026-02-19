# Backend Runtime Notes (Railway)

## Required environment variables

- `ENVIRONMENT=production`
- `DATABASE_URL=<Supabase Postgres connection string>`
- `SUPABASE_URL=https://<project-ref>.supabase.co`
- `SUPABASE_JWT_ISSUER=https://<project-ref>.supabase.co/auth/v1`
- `SUPABASE_JWT_AUDIENCE=authenticated`
- `REDIS_URL=redis://...` or `rediss://...` (Railway Redis endpoint)

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
  - `POST /trading/markets`
  - `GET /trading/markets/{slug}`
- Legacy `GET /markets` now reads from the same trading table (frontend-compatible shape).

Frontend should continue using Supabase only for auth and call FastAPI for trading data.

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
curl -i -X POST "${API}/trading/markets" \
  -H "Content-Type: application/json" \
  -d '{
    "slug":"demo-market-1",
    "title":"Will Udfall launch new mobile app this quarter?",
    "category":"finance",
    "ruleText":"Resolves YES if app is publicly released on iOS or Android before quarter end."
  }'
```

Expect: `201` and created market with `status: "draft"`.

## CI runtime check

```bash
pytest -q tests/test_db_runtime.py
```

Requires `DATABASE_URL` in CI. The test creates an engine from env and runs `SELECT 1`.
