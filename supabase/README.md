# Supabase trading migrations

This folder contains SQL migrations for the Udfall trading data model.

## Apply in Supabase SQL Editor

1. Open Supabase Dashboard -> SQL Editor.
2. Paste and run `migrations/20260219_001_trading_core.sql`.
3. Verify objects in schema `trading`.

## What this migration creates

- Schema: `trading`
- Enums for market/order/fill/wallet/ledger/resolution state
- Tables:
  - `markets`, `market_rules`, `market_state_events`
  - `orders`, `fills`, `positions`
  - `wallet_accounts`, `ledger_entries` (double-entry enforced)
  - `resolutions`, `audit_logs`
- Safety defaults:
  - Internal `bigint` identity primary keys
  - Public `uuid` IDs for API exposure
  - Money stored in integer minor units (`*_minor`)
  - Price/probability stored in integer basis points (`*_bps`)
  - UTC timestamps (`timestamptz`)
  - Anonymous/authenticated roles revoked from direct table writes

## Notes

- This migration is intentionally backend-write oriented.
- Frontend should read through backend APIs or dedicated read-only views.
- Add RLS policies in a follow-up migration once read/write boundaries are finalized.
