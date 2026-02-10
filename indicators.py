"""
Technical Indicators Engine
----------------------------
Computes a full suite of technical indicators on OHLCV DataFrames.
Uses the `ta` library where possible, with custom implementations
for composite signals.
"""

import logging

import numpy as np
import pandas as pd
import ta

import config

logger = logging.getLogger(__name__)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all technical indicators to an OHLCV DataFrame.
    Expects columns: open, high, low, close, volume.
    Returns the DataFrame with new indicator columns added.
    """
    df = df.copy()

    # ── Exponential Moving Averages ─────────────────────────────────────
    df["ema_short"] = ta.trend.ema_indicator(df["close"], window=config.EMA_SHORT)
    df["ema_long"] = ta.trend.ema_indicator(df["close"], window=config.EMA_LONG)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=config.EMA_TREND)

    # EMA crossover signal: +1 when short > long, -1 when short < long
    df["ema_crossover"] = np.where(df["ema_short"] > df["ema_long"], 1, -1)

    # ── RSI ──────────────────────────────────────────────────────────────
    df["rsi"] = ta.momentum.rsi(df["close"], window=config.RSI_PERIOD)

    # ── Stochastic RSI ───────────────────────────────────────────────────
    stoch_rsi = ta.momentum.StochRSIIndicator(
        df["close"],
        window=config.STOCH_RSI_PERIOD,
        smooth1=config.STOCH_RSI_SMOOTH_K,
        smooth2=config.STOCH_RSI_SMOOTH_D,
    )
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

    # ── MACD ─────────────────────────────────────────────────────────────
    macd = ta.trend.MACD(
        df["close"],
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ── Bollinger Bands ──────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(
        df["close"], window=config.BB_PERIOD, window_dev=config.BB_STD
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = bb.bollinger_pband()  # %B — where price sits in the band

    # ── ATR (Average True Range) ─────────────────────────────────────────
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=config.ATR_PERIOD
    )

    # ── ADX (Average Directional Index) ──────────────────────────────────
    adx = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=config.ADX_PERIOD
    )
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()  # +DI
    df["adx_neg"] = adx.adx_neg()  # -DI

    # ── OBV (On-Balance Volume) ──────────────────────────────────────────
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    # ── Volume Moving Average ────────────────────────────────────────────
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # ── VWAP (approximation for daily data) ──────────────────────────────
    if config.VWAP_ENABLED:
        df["vwap"] = _compute_vwap(df)

    # ── Rate of Change ───────────────────────────────────────────────────
    df["roc_10"] = ta.momentum.roc(df["close"], window=10)

    # ── Support / Resistance levels (rolling) ────────────────────────────
    df["resistance"] = df["high"].rolling(window=20).max()
    df["support"] = df["low"].rolling(window=20).min()

    logger.info(f"Computed indicators — {len(df.columns)} total columns")
    return df


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP approximation."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Extract the most recent row's indicator values as a dict."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        "close": last.get("close"),
        "ema_crossover": last.get("ema_crossover"),
        "rsi": last.get("rsi"),
        "macd_hist": last.get("macd_hist"),
        "bb_pct": last.get("bb_pct"),
        "adx": last.get("adx"),
        "atr": last.get("atr"),
        "volume_ratio": last.get("volume_ratio"),
        "roc_10": last.get("roc_10"),
        "stoch_rsi_k": last.get("stoch_rsi_k"),
    }
