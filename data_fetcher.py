"""
Data Fetcher
-------------
Downloads and caches market data from Yahoo Finance (free, no API key).
Provides OHLCV data with automatic retry and rate-limit handling.
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


def fetch_ticker_data(
    ticker: str,
    period: str = config.DATA_PERIOD,
    interval: str = config.DATA_INTERVAL,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a single ticker with retry logic."""
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            df.index = pd.to_datetime(df.index)
            # Standardise column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["ticker"] = ticker
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            wait = 2 ** attempt
            logger.error(f"Error fetching {ticker} (attempt {attempt+1}): {e}")
            time.sleep(wait)
    return None


def fetch_all(
    tickers: Optional[List[str]] = None,
    period: str = config.DATA_PERIOD,
    interval: str = config.DATA_INTERVAL,
) -> Dict[str, pd.DataFrame]:
    """Fetch data for all watchlist tickers. Returns {ticker: DataFrame}."""
    tickers = tickers or config.WATCHLIST
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = fetch_ticker_data(ticker, period=period, interval=interval)
        if df is not None:
            data[ticker] = df
    logger.info(f"Fetched data for {len(data)}/{len(tickers)} tickers")
    return data


def fetch_realtime_quote(ticker: str) -> Optional[dict]:
    """Get the latest quote info for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            "ticker": ticker,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        logger.error(f"Error fetching quote for {ticker}: {e}")
        return None
