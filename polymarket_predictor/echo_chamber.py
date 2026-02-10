"""
Echo Chamber Detector & Narrative Momentum Tracker
----------------------------------------------------
Two critical modules that most sentiment bots miss:

1. ECHO CHAMBER DETECTOR
   When 90% of Reddit/Twitter agrees on something, it's usually
   already priced in. The detector identifies when sentiment is
   suspiciously uniform and either:
   - Penalizes the signal (the crowd is already in the price)
   - Generates a contrarian signal (the crowd is probably wrong)

2. NARRATIVE MOMENTUM TRACKER
   Tracks HOW sentiment is evolving, not just where it is.
   Key signals:
   - Sentiment acceleration: sentiment changing faster and faster
   - Narrative phase: early (few voices, high engagement) vs.
     late (everyone talking, declining engagement)
   - Source divergence: when Reddit and Twitter disagree, who's right?
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class EchoChamberReading:
    """Result of echo chamber analysis for a market."""
    is_echo_chamber: bool          # True if suspiciously uniform
    uniformity_score: float        # 0 = diverse, 1 = total agreement
    dominant_direction: str        # "YES" or "NO"
    dominant_pct: float            # % agreeing with dominant direction
    contrarian_signal: bool        # True if we should fade the crowd
    confidence_adjustment: float   # Multiply signal confidence by this
    source_divergence: float       # 0 = sources agree, 1 = sources disagree
    explanation: str


@dataclass
class NarrativeMomentum:
    """Measures how sentiment is changing over time."""
    velocity: float                # Rate of change (positive = becoming more YES)
    acceleration: float            # Is the velocity increasing or decreasing?
    phase: str                     # "early", "building", "peak", "fading"
    momentum_score: float          # -1 to +1, directional momentum
    volume_trend: str              # "rising", "stable", "falling"
    hours_since_shift: float       # Hours since last significant direction change
    explanation: str


class EchoChamberDetector:
    """
    Detects when sentiment is too uniform (echo chamber) and generates
    contrarian signals when appropriate.

    Key insight: In prediction markets, when public sentiment overwhelmingly
    agrees, the market price usually already reflects this. The remaining
    edge is in identifying when the crowd is WRONG.
    """

    def analyze(
        self,
        sentiment_data: list,  # List of SentimentResult
        source_breakdown: Dict[str, float],
        bullish_pct: float,
        bearish_pct: float,
        sample_size: int,
    ) -> EchoChamberReading:
        """
        Analyze sentiment distribution for echo chamber patterns.
        """
        if sample_size < config.MIN_SAMPLE_SIZE:
            return EchoChamberReading(
                is_echo_chamber=False,
                uniformity_score=0.0,
                dominant_direction="NEUTRAL",
                dominant_pct=0.0,
                contrarian_signal=False,
                confidence_adjustment=1.0,
                source_divergence=0.0,
                explanation="Insufficient data for echo chamber analysis",
            )

        # ── Uniformity Score ────────────────────────────────────────
        # How much does the crowd agree?
        dominant_pct = max(bullish_pct, bearish_pct)
        dominant_dir = "YES" if bullish_pct > bearish_pct else "NO"
        neutral_pct = 1.0 - bullish_pct - bearish_pct

        # Uniformity: 1.0 = everyone agrees, 0.0 = perfect split
        uniformity = dominant_pct - (1.0 - dominant_pct - neutral_pct)
        uniformity = max(uniformity, 0.0)

        # ── Source Divergence ───────────────────────────────────────
        # Do Reddit and Twitter agree?
        source_values = list(source_breakdown.values())
        if len(source_values) >= 2:
            source_div = float(np.std(source_values)) * 2  # Scale to 0-1 ish
            source_div = min(source_div, 1.0)
        else:
            source_div = 0.0

        # ── Echo Chamber Detection ──────────────────────────────────
        is_echo = uniformity > config.ECHO_CHAMBER_THRESHOLD

        # ── Confidence Adjustment ───────────────────────────────────
        if is_echo:
            # Echo chamber detected — reduce confidence in the dominant signal
            conf_adj = config.ECHO_CHAMBER_PENALTY
            explanation = (
                f"ECHO CHAMBER: {dominant_pct:.0%} agree on {dominant_dir}. "
                f"Signal likely already priced in. Confidence reduced by "
                f"{(1 - conf_adj):.0%}."
            )
        elif source_div > 0.3:
            # Sources disagree — interesting signal but lower confidence
            conf_adj = 0.8
            explanation = (
                f"SOURCE DIVERGENCE: Reddit and Twitter disagree "
                f"(divergence={source_div:.2f}). Mixed signal."
            )
        elif uniformity > 0.6:
            # Moderate agreement — slight penalty
            conf_adj = 0.9
            explanation = (
                f"Moderate agreement ({dominant_pct:.0%} {dominant_dir}). "
                f"Slight confidence reduction."
            )
        else:
            # Healthy diversity of opinion — signal is more meaningful
            conf_adj = 1.1  # Small boost
            explanation = (
                f"Healthy opinion diversity (uniformity={uniformity:.2f}). "
                f"Signal has informational value."
            )

        # ── Contrarian Signal ───────────────────────────────────────
        # When the echo chamber is extreme AND sources diverge,
        # the minority might be the smart money
        contrarian = (
            is_echo
            and source_div > 0.2
            and sample_size >= config.MIN_SAMPLE_SIZE * 2
        )
        if contrarian:
            conf_adj *= config.CONTRARIAN_BOOST
            explanation += " CONTRARIAN opportunity detected."

        return EchoChamberReading(
            is_echo_chamber=is_echo,
            uniformity_score=float(uniformity),
            dominant_direction=dominant_dir,
            dominant_pct=float(dominant_pct),
            contrarian_signal=contrarian,
            confidence_adjustment=float(conf_adj),
            source_divergence=float(source_div),
            explanation=explanation,
        )


class NarrativeTracker:
    """
    Tracks how sentiment evolves over time.

    Rather than just measuring "what is sentiment now?", this measures:
    - Velocity: is sentiment shifting toward YES or NO?
    - Acceleration: is the shift speeding up or slowing down?
    - Phase: is this narrative early (valuable) or late (priced in)?
    """

    def analyze(
        self,
        time_series: List[Tuple[datetime, float]],
        current_sentiment: float,
    ) -> NarrativeMomentum:
        """
        Analyze the trajectory of sentiment for a market.

        time_series: [(timestamp, yes_probability), ...]
        """
        if len(time_series) < 3:
            return NarrativeMomentum(
                velocity=0.0, acceleration=0.0, phase="insufficient",
                momentum_score=0.0, volume_trend="unknown",
                hours_since_shift=0.0,
                explanation="Not enough time-series data",
            )

        now = datetime.now(timezone.utc)
        sorted_ts = sorted(time_series, key=lambda x: x[0])

        # Split into time windows
        recent = []    # Last VELOCITY_WINDOW hours
        older = []     # VELOCITY_WINDOW to 2x VELOCITY_WINDOW hours ago
        oldest = []    # Everything older

        for ts, val in sorted_ts:
            age_hours = (now - ts).total_seconds() / 3600
            if age_hours <= config.VELOCITY_WINDOW_HOURS:
                recent.append(val)
            elif age_hours <= config.VELOCITY_WINDOW_HOURS * 2:
                older.append(val)
            else:
                oldest.append(val)

        # ── Velocity: rate of sentiment change ──────────────────────
        recent_avg = np.mean(recent) if recent else current_sentiment
        older_avg = np.mean(older) if older else recent_avg

        velocity = float(recent_avg - older_avg)

        # ── Acceleration: is velocity increasing? ───────────────────
        oldest_avg = np.mean(oldest) if oldest else older_avg
        prev_velocity = float(older_avg - oldest_avg)
        acceleration = velocity - prev_velocity

        # ── Narrative Phase ─────────────────────────────────────────
        volume_recent = len(recent)
        volume_older = max(len(older), 1)
        volume_ratio = volume_recent / volume_older

        if volume_ratio > 2.0 and abs(velocity) > 0.05:
            phase = "building"    # More people talking, sentiment shifting
        elif volume_ratio > 1.5 and abs(velocity) < 0.03:
            phase = "peak"        # Everyone's talked about it, settling
        elif volume_ratio < 0.5:
            phase = "fading"      # Interest dying down
        elif len(sorted_ts) < 10:
            phase = "early"       # Few data points = early narrative
        else:
            phase = "stable"

        # Volume trend
        if volume_ratio > 1.3:
            vol_trend = "rising"
        elif volume_ratio < 0.7:
            vol_trend = "falling"
        else:
            vol_trend = "stable"

        # ── Momentum Score (-1 to +1) ───────────────────────────────
        # Combines velocity + acceleration + phase
        momentum = velocity * 3  # Scale up velocity

        # Boost momentum in building phase, dampen in peak/fading
        phase_mult = {
            "early": 1.5,      # Early signals are most valuable
            "building": 1.3,
            "stable": 1.0,
            "peak": 0.5,       # Already priced in
            "fading": 0.3,
        }.get(phase, 1.0)

        momentum *= phase_mult

        # Acceleration bonus
        if (velocity > 0 and acceleration > 0) or (velocity < 0 and acceleration < 0):
            momentum *= 1.2  # Accelerating in same direction

        momentum = float(np.clip(momentum, -1.0, 1.0))

        # ── Hours Since Last Direction Change ───────────────────────
        hours_since_shift = self._find_last_shift(sorted_ts, now)

        explanation = (
            f"Phase: {phase} | Velocity: {velocity:+.3f}/window | "
            f"Acceleration: {acceleration:+.3f} | Volume: {vol_trend} | "
            f"Momentum: {momentum:+.2f}"
        )

        return NarrativeMomentum(
            velocity=float(velocity),
            acceleration=float(acceleration),
            phase=phase,
            momentum_score=momentum,
            volume_trend=vol_trend,
            hours_since_shift=hours_since_shift,
            explanation=explanation,
        )

    def _find_last_shift(
        self,
        sorted_ts: List[Tuple[datetime, float]],
        now: datetime,
    ) -> float:
        """Find hours since the last significant sentiment direction change."""
        if len(sorted_ts) < 5:
            return 0.0

        # Compute rolling average and find when it crossed 0.5
        window = 3
        for i in range(len(sorted_ts) - window, 0, -1):
            chunk = [v for _, v in sorted_ts[i : i + window]]
            prev_chunk = [v for _, v in sorted_ts[max(0, i - window) : i]]

            if not chunk or not prev_chunk:
                continue

            avg_now = np.mean(chunk)
            avg_prev = np.mean(prev_chunk)

            # Direction change: crossed the 0.5 line
            if (avg_now > 0.5 and avg_prev < 0.5) or (avg_now < 0.5 and avg_prev > 0.5):
                shift_time = sorted_ts[i][0]
                return (now - shift_time).total_seconds() / 3600

        return float("inf")
