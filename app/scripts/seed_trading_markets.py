import json
import os
from datetime import datetime, timedelta, timezone

from app.storage import (
    SessionLocal,
    TradingMarket,
    TradingMarketCategory,
    TradingMarketRule,
    TradingMarketStateEvent,
    TradingMarketStatus,
)

SEED_MARKETS = [
    {
        "slug": "politics_us_president_2028",
        "title": "Will the Democratic candidate win the US presidential election 2028?",
        "description": "Resolves YES if the Democratic nominee wins the 2028 US presidential election.",
        "category": TradingMarketCategory.politics,
    },
    {
        "slug": "politics_eu_parliament_turnout_2029",
        "title": "Will EU Parliament turnout exceed 55 percent in 2029?",
        "description": "Resolves YES if official EU turnout is greater than 55.0%.",
        "category": TradingMarketCategory.politics,
    },
    {
        "slug": "sports_nba_champion_2027_lakers",
        "title": "Will the Lakers win the NBA championship 2027 season?",
        "description": "Resolves YES if the Los Angeles Lakers win the NBA Finals for the 2026-2027 season.",
        "category": TradingMarketCategory.sports,
    },
    {
        "slug": "sports_champions_league_2027_real_madrid",
        "title": "Will Real Madrid win the 2027 UEFA Champions League?",
        "description": "Resolves YES if Real Madrid are official tournament winners.",
        "category": TradingMarketCategory.sports,
    },
    {
        "slug": "finance_btc_over_150k_2027",
        "title": "Will Bitcoin trade above 150000 USD before 2027-12-31?",
        "description": "Resolves YES if BTC/USD spot price is greater than 150000 on a major exchange before deadline.",
        "category": TradingMarketCategory.finance,
    },
    {
        "slug": "finance_sp500_over_7000_2028",
        "title": "Will S and P 500 close above 7000 before 2028-12-31?",
        "description": "Resolves YES if official S and P 500 close exceeds 7000.00 before the deadline.",
        "category": TradingMarketCategory.finance,
    },
    {
        "slug": "entertainment_oscars_2027_top_grossing_wins",
        "title": "Will the top grossing film of 2026 win Best Picture at 2027 Oscars?",
        "description": "Resolves YES if the highest grossing worldwide 2026 release wins Best Picture in 2027.",
        "category": TradingMarketCategory.entertainment,
    },
    {
        "slug": "entertainment_streaming_netflix_subs_up_2027",
        "title": "Will Netflix report subscriber growth in every quarter of 2027?",
        "description": "Resolves YES if each quarterly report in calendar year 2027 shows net additions above zero.",
        "category": TradingMarketCategory.entertainment,
    },
    {
        "slug": "finance_tesla_market_cap_over_2t_2028",
        "title": "Will Tesla market cap exceed 2 trillion USD before 2028-12-31?",
        "description": "Resolves YES if Tesla market capitalization is greater than 2T USD at any regular close before deadline.",
        "category": TradingMarketCategory.finance,
    },
    {
        "slug": "sports_fifa_world_cup_2026_brazil_champion",
        "title": "Will Brazil win FIFA World Cup 2026?",
        "description": "Resolves YES if Brazil are official FIFA World Cup 2026 winners.",
        "category": TradingMarketCategory.sports,
    },
]


def _rules_payload(title: str) -> dict:
    return {
        "version": 1,
        "resolution_source": "official_result",
        "rule_text": title,
        "disputes": {"window_seconds": 86400},
    }


def main() -> None:
    environment = (os.getenv("ENVIRONMENT") or "development").lower()
    is_production = environment == "production"
    allow_production = os.getenv("ALLOW_PRODUCTION_SEED", "false").lower() == "true"
    allow_full_production_seed = os.getenv("ALLOW_PRODUCTION_FULL_SEED", "false").lower() == "true"
    seed_limit_raw = os.getenv("SEED_MARKET_LIMIT")

    if is_production and not allow_production:
        raise RuntimeError(
            "Seeding is blocked in production. Set ALLOW_PRODUCTION_SEED=true to override intentionally."
        )

    seed_markets = SEED_MARKETS
    if is_production and not allow_full_production_seed:
        # Default production safety rail: seed only two markets unless explicitly overridden.
        seed_markets = SEED_MARKETS[:2]

    if seed_limit_raw:
        try:
            seed_limit = int(seed_limit_raw)
        except ValueError as exc:
            raise RuntimeError("SEED_MARKET_LIMIT must be a positive integer") from exc
        if seed_limit <= 0:
            raise RuntimeError("SEED_MARKET_LIMIT must be a positive integer")
        seed_markets = seed_markets[:seed_limit]

    db = SessionLocal()
    now = datetime.now(timezone.utc)
    created = 0
    updated = 0
    try:
        for index, item in enumerate(seed_markets):
            open_time = now + timedelta(hours=index)
            close_time = now + timedelta(days=21 + index)
            resolve_time = close_time + timedelta(hours=6)

            market = (
                db.query(TradingMarket)
                .filter(TradingMarket.slug == item["slug"])
                .one_or_none()
            )
            is_new = market is None
            if market is None:
                market = TradingMarket(
                    slug=item["slug"],
                    title=item["title"],
                    description=item["description"],
                    category=item["category"],
                    status=TradingMarketStatus.live,
                    yes_price_bps=5000,
                    tick_size_bps=1,
                    min_order_size_minor=100,
                    max_order_size_minor=500000000,
                    open_time=open_time,
                    close_time=close_time,
                    resolve_time=resolve_time,
                    trading_starts_at=open_time,
                    trading_ends_at=close_time,
                    resolves_after=resolve_time,
                    created_at=now,
                    updated_at=now,
                )
                db.add(market)
                created += 1
            else:
                market.title = item["title"]
                market.description = item["description"]
                market.category = item["category"]
                market.status = TradingMarketStatus.live
                market.open_time = open_time
                market.close_time = close_time
                market.resolve_time = resolve_time
                market.trading_starts_at = open_time
                market.trading_ends_at = close_time
                market.resolves_after = resolve_time
                market.updated_at = now
                updated += 1

            db.flush()

            rules_json = _rules_payload(item["title"])
            rule = (
                db.query(TradingMarketRule)
                .filter(
                    TradingMarketRule.market_id == market.id,
                    TradingMarketRule.is_active.is_(True),
                )
                .order_by(TradingMarketRule.version.desc())
                .first()
            )
            if rule is None:
                db.add(
                    TradingMarketRule(
                        market_id=market.id,
                        version=1,
                        rule_text=json.dumps(rules_json, sort_keys=True, separators=(",", ":")),
                        rules_json=rules_json,
                        is_active=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                rule.rule_text = json.dumps(rules_json, sort_keys=True, separators=(",", ":"))
                rule.rules_json = rules_json
                rule.updated_at = now

            db.add(
                TradingMarketStateEvent(
                    market_id=market.id,
                    from_status=None if is_new else TradingMarketStatus.live,
                    to_status=TradingMarketStatus.live,
                    reason="seed_created" if is_new else "seed_updated",
                    event_metadata={"seed": True, "slug": market.slug},
                    created_at=now,
                )
            )

        db.commit()
        print(f"seed completed created={created} updated={updated} total={len(seed_markets)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
