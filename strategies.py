"""
Trading Strategies
-------------------
Each strategy consumes indicator-enriched DataFrames and returns a
signal dict with:
    direction : "BUY" | "SELL" | "HOLD"
    confidence: float 0-1 (how strong the signal is)
    reason    : str   (human-readable explanation)

The signal aggregator in `engine.py` combines these with configurable
weights to produce a final trading decision.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: str   # BUY, SELL, HOLD
    confidence: float  # 0.0 – 1.0
    reason: str
    strategy: str


# ════════════════════════════════════════════════════════════════════════
#  1. MOMENTUM STRATEGY
#     Buys when multiple momentum indicators align in the same direction.
#     Uses EMA crossover + MACD histogram + RSI + ROC confirmation.
# ════════════════════════════════════════════════════════════════════════

def momentum_strategy(df: pd.DataFrame) -> Signal:
    """Multi-factor momentum: needs confluence of several indicators."""
    if len(df) < config.EMA_TREND:
        return Signal("HOLD", 0.0, "Insufficient data", "momentum")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    reasons = []

    # Factor 1: EMA crossover (short > long AND price above trend EMA)
    if last["ema_short"] > last["ema_long"] and last["close"] > last["ema_trend"]:
        score += 0.25
        reasons.append("EMA bullish crossover above trend")
    elif last["ema_short"] < last["ema_long"] and last["close"] < last["ema_trend"]:
        score -= 0.25
        reasons.append("EMA bearish crossover below trend")

    # Factor 2: MACD histogram turning positive / negative
    if last["macd_hist"] > 0 and prev["macd_hist"] <= 0:
        score += 0.25
        reasons.append("MACD histogram turned positive")
    elif last["macd_hist"] < 0 and prev["macd_hist"] >= 0:
        score -= 0.25
        reasons.append("MACD histogram turned negative")
    elif last["macd_hist"] > 0:
        score += 0.10
    elif last["macd_hist"] < 0:
        score -= 0.10

    # Factor 3: RSI momentum (not overbought/oversold extremes)
    rsi = last["rsi"]
    if 50 < rsi < config.RSI_OVERBOUGHT:
        score += 0.20
        reasons.append(f"RSI bullish ({rsi:.0f})")
    elif config.RSI_OVERSOLD < rsi < 50:
        score -= 0.20
        reasons.append(f"RSI bearish ({rsi:.0f})")

    # Factor 4: Rate of change positive
    if last["roc_10"] > 2:
        score += 0.15
        reasons.append(f"Strong upward ROC ({last['roc_10']:.1f}%)")
    elif last["roc_10"] < -2:
        score -= 0.15
        reasons.append(f"Strong downward ROC ({last['roc_10']:.1f}%)")

    # Factor 5: Volume confirmation
    if last["volume_ratio"] > 1.5:
        score *= 1.2  # Amplify signal on high volume
        reasons.append("High volume confirmation")

    # Convert score to signal
    confidence = min(abs(score), 1.0)
    if score > 0.15:
        return Signal("BUY", confidence, " | ".join(reasons), "momentum")
    elif score < -0.15:
        return Signal("SELL", confidence, " | ".join(reasons), "momentum")
    return Signal("HOLD", confidence, "No clear momentum", "momentum")


# ════════════════════════════════════════════════════════════════════════
#  2. MEAN REVERSION STRATEGY
#     Buys oversold conditions, sells overbought.
#     Uses Bollinger Bands %B + RSI + Stochastic RSI.
# ════════════════════════════════════════════════════════════════════════

def mean_reversion_strategy(df: pd.DataFrame) -> Signal:
    """Buy oversold / sell overbought using BB + RSI + StochRSI."""
    if len(df) < config.BB_PERIOD + 5:
        return Signal("HOLD", 0.0, "Insufficient data", "mean_reversion")

    last = df.iloc[-1]
    score = 0.0
    reasons = []

    # Bollinger Band %B: < 0 means below lower band, > 1 means above upper
    bb_pct = last["bb_pct"]
    if bb_pct < 0.05:
        score += 0.35
        reasons.append(f"Price near lower Bollinger Band (%B={bb_pct:.2f})")
    elif bb_pct > 0.95:
        score -= 0.35
        reasons.append(f"Price near upper Bollinger Band (%B={bb_pct:.2f})")

    # RSI extremes
    rsi = last["rsi"]
    if rsi < config.RSI_OVERSOLD:
        score += 0.30
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > config.RSI_OVERBOUGHT:
        score -= 0.30
        reasons.append(f"RSI overbought ({rsi:.0f})")

    # Stochastic RSI extremes
    stoch_k = last["stoch_rsi_k"]
    if stoch_k is not None and not np.isnan(stoch_k):
        if stoch_k < 0.20:
            score += 0.20
            reasons.append(f"StochRSI oversold ({stoch_k:.2f})")
        elif stoch_k > 0.80:
            score -= 0.20
            reasons.append(f"StochRSI overbought ({stoch_k:.2f})")

    # Require ADX < threshold (ranging market favors mean reversion)
    adx = last["adx"]
    if adx is not None and not np.isnan(adx):
        if adx < config.ADX_THRESHOLD:
            score *= 1.3  # Boost signal in low-trend environments
            reasons.append(f"Low ADX ({adx:.0f}) — ranging market")
        else:
            score *= 0.5  # Dampen signal in trending markets
            reasons.append(f"High ADX ({adx:.0f}) — trending, risky for MR")

    confidence = min(abs(score), 1.0)
    if score > 0.20:
        return Signal("BUY", confidence, " | ".join(reasons), "mean_reversion")
    elif score < -0.20:
        return Signal("SELL", confidence, " | ".join(reasons), "mean_reversion")
    return Signal("HOLD", confidence, "No mean-reversion signal", "mean_reversion")


# ════════════════════════════════════════════════════════════════════════
#  3. BREAKOUT STRATEGY
#     Detects price breaking above resistance or below support with
#     volume confirmation and volatility expansion.
# ════════════════════════════════════════════════════════════════════════

def breakout_strategy(df: pd.DataFrame) -> Signal:
    """Breakout detection: price vs support/resistance + volume surge."""
    if len(df) < 25:
        return Signal("HOLD", 0.0, "Insufficient data", "breakout")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    reasons = []

    close = last["close"]
    resistance = last["resistance"]
    support = last["support"]

    # Breakout above resistance
    if close > resistance and prev["close"] <= prev["resistance"]:
        score += 0.40
        reasons.append(f"Breakout above resistance ({resistance:.2f})")
    # Breakdown below support
    elif close < support and prev["close"] >= prev["support"]:
        score -= 0.40
        reasons.append(f"Breakdown below support ({support:.2f})")

    # Volume surge confirmation (volume > 2x average)
    vol_ratio = last["volume_ratio"]
    if vol_ratio > 2.0:
        score *= 1.4
        reasons.append(f"Volume surge ({vol_ratio:.1f}x average)")
    elif vol_ratio > 1.5:
        score *= 1.2
        reasons.append(f"Above-average volume ({vol_ratio:.1f}x)")

    # Bollinger Band width expansion (volatility expanding)
    if len(df) > 5:
        recent_widths = df["bb_width"].tail(5)
        if last["bb_width"] > recent_widths.mean() * 1.2:
            score *= 1.15
            reasons.append("Bollinger width expanding")

    # ADX rising confirms trend is forming
    adx = last["adx"]
    if adx is not None and not np.isnan(adx) and adx > config.ADX_THRESHOLD:
        score *= 1.2
        reasons.append(f"ADX confirming trend ({adx:.0f})")

    confidence = min(abs(score), 1.0)
    if score > 0.20:
        return Signal("BUY", confidence, " | ".join(reasons), "breakout")
    elif score < -0.20:
        return Signal("SELL", confidence, " | ".join(reasons), "breakout")
    return Signal("HOLD", confidence, "No breakout detected", "breakout")


# ════════════════════════════════════════════════════════════════════════
#  4. TREND FOLLOWING STRATEGY
#     Rides established trends using ADX + DI crossover + EMA alignment.
#     Only enters in the direction of the trend and uses trailing stops.
# ════════════════════════════════════════════════════════════════════════

def trend_follow_strategy(df: pd.DataFrame) -> Signal:
    """Trend following: ADX-confirmed directional moves with EMA alignment."""
    if len(df) < config.EMA_TREND + 5:
        return Signal("HOLD", 0.0, "Insufficient data", "trend_follow")

    last = df.iloc[-1]
    score = 0.0
    reasons = []

    adx = last["adx"]
    plus_di = last["adx_pos"]
    minus_di = last["adx_neg"]

    # Require strong trend (ADX above threshold)
    if adx is None or np.isnan(adx) or adx < config.ADX_THRESHOLD:
        return Signal("HOLD", 0.0, f"Weak trend (ADX={adx})", "trend_follow")

    reasons.append(f"Strong trend (ADX={adx:.0f})")

    # Directional Index crossover
    if plus_di > minus_di:
        score += 0.30
        reasons.append(f"+DI ({plus_di:.0f}) > -DI ({minus_di:.0f})")
    else:
        score -= 0.30
        reasons.append(f"-DI ({minus_di:.0f}) > +DI ({plus_di:.0f})")

    # EMA alignment (short > long > trend = strong uptrend)
    if last["ema_short"] > last["ema_long"] > last["ema_trend"]:
        score += 0.30
        reasons.append("Perfect EMA bullish alignment")
    elif last["ema_short"] < last["ema_long"] < last["ema_trend"]:
        score -= 0.30
        reasons.append("Perfect EMA bearish alignment")

    # OBV trend confirmation
    if len(df) >= 10:
        obv_recent = df["obv"].tail(10)
        obv_slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / max(abs(obv_recent.iloc[0]), 1)
        if score > 0 and obv_slope > 0:
            score += 0.15
            reasons.append("OBV confirming uptrend")
        elif score < 0 and obv_slope < 0:
            score -= 0.15
            reasons.append("OBV confirming downtrend")

    confidence = min(abs(score), 1.0)
    if score > 0.20:
        return Signal("BUY", confidence, " | ".join(reasons), "trend_follow")
    elif score < -0.20:
        return Signal("SELL", confidence, " | ".join(reasons), "trend_follow")
    return Signal("HOLD", confidence, "No clear trend signal", "trend_follow")


# ════════════════════════════════════════════════════════════════════════
#  Registry — maps strategy names to their functions
# ════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    "momentum": momentum_strategy,
    "mean_reversion": mean_reversion_strategy,
    "breakout": breakout_strategy,
    "trend_follow": trend_follow_strategy,
}
