"""
Polymarket API Client
----------------------
Fetches live markets, prices, order books, and volume data from
Polymarket's public CLOB and Gamma APIs.

No API key needed for reading — only for placing actual bets.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

import config

logger = logging.getLogger(__name__)

# Polymarket uses two APIs:
#   Gamma API — market metadata, search, event info
#   CLOB API  — prices, order books, trades


def fetch_active_markets(
    limit: int = 100,
    min_liquidity: float = config.MIN_MARKET_LIQUIDITY,
) -> List[dict]:
    """
    Fetch active prediction markets from Polymarket.
    Returns a list of market dicts with prices, volume, and metadata.
    """
    markets = []

    for attempt in range(3):
        try:
            # Gamma API for market discovery
            resp = requests.get(
                f"{config.POLYMARKET_GAMMA_API}/markets",
                params={
                    "limit": limit,
                    "active": True,
                    "closed": False,
                    "order": "volume24hr",
                    "ascending": False,
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw_markets = resp.json()

            for m in raw_markets:
                market = _parse_market(m)
                if market and market["liquidity"] >= min_liquidity:
                    markets.append(market)

            logger.info(f"Fetched {len(markets)} active markets from Polymarket")
            return markets

        except Exception as e:
            wait = 2 ** attempt
            logger.error(f"Error fetching markets (attempt {attempt+1}): {e}")
            time.sleep(wait)

    return markets


def fetch_market_by_id(condition_id: str) -> Optional[dict]:
    """Fetch a single market by its condition ID."""
    try:
        resp = requests.get(
            f"{config.POLYMARKET_GAMMA_API}/markets/{condition_id}",
            timeout=10,
        )
        resp.raise_for_status()
        return _parse_market(resp.json())
    except Exception as e:
        logger.error(f"Error fetching market {condition_id}: {e}")
        return None


def fetch_market_prices(token_id: str) -> Optional[dict]:
    """Fetch current best bid/ask from the CLOB for a token."""
    try:
        resp = requests.get(
            f"{config.POLYMARKET_API_BASE}/book",
            params={"token_id": token_id},
            timeout=10,
        )
        resp.raise_for_status()
        book = resp.json()

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        mid_price = (best_bid + best_ask) / 2

        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])

        return {
            "token_id": token_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": best_ask - best_bid,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
        }
    except Exception as e:
        logger.error(f"Error fetching prices for {token_id}: {e}")
        return None


def search_markets(query: str, limit: int = 20) -> List[dict]:
    """Search markets by keyword."""
    try:
        resp = requests.get(
            f"{config.POLYMARKET_GAMMA_API}/markets",
            params={
                "limit": limit,
                "active": True,
                "closed": False,
                "tag": query,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return [_parse_market(m) for m in resp.json() if _parse_market(m)]
    except Exception as e:
        logger.error(f"Error searching markets for '{query}': {e}")
        return []


def fetch_market_history(condition_id: str, fidelity: int = 60) -> List[dict]:
    """
    Fetch price history for a market.
    fidelity = minutes between data points (60 = hourly).
    """
    try:
        resp = requests.get(
            f"{config.POLYMARKET_GAMMA_API}/markets/{condition_id}/timeseries",
            params={"fidelity": fidelity},
            timeout=10,
        )
        resp.raise_for_status()
        history = resp.json()
        return [
            {
                "timestamp": point.get("t"),
                "price": float(point.get("p", 0)),
            }
            for point in history
        ]
    except Exception as e:
        logger.error(f"Error fetching history for {condition_id}: {e}")
        return []


def get_market_categories() -> List[str]:
    """Return the major market categories on Polymarket."""
    return [
        "politics", "crypto", "sports", "pop-culture",
        "science", "business", "world", "tech",
    ]


# ── Helpers ─────────────────────────────────────────────────────────────

def _parse_market(raw: dict) -> Optional[dict]:
    """Normalize raw API market data into our standard format."""
    if not raw:
        return None

    try:
        # Extract YES price (probability)
        outcomes = raw.get("outcomes", ["Yes", "No"])
        outcome_prices = raw.get("outcomePrices", [])

        if outcome_prices and len(outcome_prices) >= 1:
            yes_price = float(outcome_prices[0])
        elif raw.get("bestBid"):
            yes_price = float(raw["bestBid"])
        else:
            yes_price = 0.5

        # Parse dates
        end_date = raw.get("endDate") or raw.get("end_date_iso")
        days_to_resolution = None
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_to_resolution = (end_dt - now).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

        volume_24h = float(raw.get("volume24hr", 0) or 0)
        total_volume = float(raw.get("volumeNum", 0) or raw.get("volume", 0) or 0)
        liquidity = float(raw.get("liquidity", 0) or 0)

        return {
            "id": raw.get("id") or raw.get("condition_id", ""),
            "condition_id": raw.get("conditionId") or raw.get("condition_id", ""),
            "question": raw.get("question", ""),
            "description": raw.get("description", "")[:500],
            "category": raw.get("groupItemTitle") or raw.get("category", ""),
            "yes_price": yes_price,
            "no_price": 1.0 - yes_price,
            "outcomes": outcomes,
            "volume_24h": volume_24h,
            "total_volume": total_volume,
            "liquidity": liquidity,
            "end_date": end_date,
            "days_to_resolution": days_to_resolution,
            "image": raw.get("image", ""),
            "slug": raw.get("slug", ""),
            "tokens": raw.get("clobTokenIds", []),
        }
    except Exception as e:
        logger.debug(f"Error parsing market: {e}")
        return None


def filter_markets(
    markets: List[dict],
    min_volume: float = config.MIN_MARKET_VOLUME,
    max_days: float = config.MAX_DAYS_TO_RESOLUTION,
    min_days: float = config.MIN_DAYS_TO_RESOLUTION,
) -> List[dict]:
    """Apply filters to select tradeable markets."""
    filtered = []
    for m in markets:
        days = m.get("days_to_resolution")
        vol = m.get("volume_24h", 0)

        # Must have enough volume
        if vol < min_volume:
            continue

        # Must resolve within our window (if we know the date)
        if days is not None:
            if days > max_days or days < min_days:
                continue

        # Skip markets already near certainty (no edge possible)
        yes_p = m.get("yes_price", 0.5)
        if yes_p < 0.03 or yes_p > 0.97:
            continue

        filtered.append(m)

    logger.info(f"Filtered to {len(filtered)}/{len(markets)} tradeable markets")
    return filtered
