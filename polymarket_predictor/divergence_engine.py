"""
Sentiment-Price Divergence Engine
-----------------------------------
THIS IS THE CORE ALPHA GENERATOR.

The edge: when public sentiment is strongly shifting in one direction
but the Polymarket price hasn't caught up yet.

Types of divergence:
  1. LEVEL DIVERGENCE
     Sentiment says 70% YES, market price is 55% YES.
     -> The crowd knows something the market hasn't absorbed.

  2. VELOCITY DIVERGENCE
     Sentiment is rapidly shifting toward YES, but price is flat.
     -> The price will likely follow sentiment with a lag.

  3. CONTRARIAN DIVERGENCE (most profitable, highest risk)
     Sentiment is 90% YES (echo chamber), but the market is 60%.
     -> The smart money is fading the crowd.
     -> If the echo chamber is wrong, we profit from the correction.

  4. CROSS-PLATFORM DIVERGENCE
     Reddit says 65% YES, Twitter says 45% YES.
     -> Information is fragmented. One platform knows more.
     -> Usually the platform with the subject-matter experts wins.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import config
from polymarket_predictor.bayesian_updater import BayesianEstimate
from polymarket_predictor.echo_chamber import EchoChamberReading, NarrativeMomentum
from polymarket_predictor.sentiment_analyzer import AggregatedSentiment

logger = logging.getLogger(__name__)


@dataclass
class DivergenceSignal:
    """A detected divergence between sentiment and market price."""
    market_id: str
    question: str
    signal_type: str           # "level", "velocity", "contrarian", "cross_platform"
    direction: str             # "BUY_YES" or "BUY_NO"
    edge_estimate: float       # Estimated edge (e.g., 0.12 = 12%)
    confidence: float          # 0-1
    market_price: float        # Current YES price
    sentiment_price: float     # Sentiment-implied YES price
    bayesian_posterior: float  # Bayesian estimate
    strength: str              # "strong", "moderate", "weak"
    explanation: str


class DivergenceEngine:
    """
    Detects and scores divergences between sentiment and Polymarket prices.
    This is where the money is made.
    """

    def analyze(
        self,
        market: dict,
        sentiment: AggregatedSentiment,
        bayesian: BayesianEstimate,
        echo: EchoChamberReading,
        narrative: NarrativeMomentum,
    ) -> Optional[DivergenceSignal]:
        """
        Full divergence analysis combining all signal components.
        Returns None if no actionable divergence is found.
        """
        market_price = market["yes_price"]
        question = market["question"]
        market_id = market["id"]

        # ── 1. Level Divergence ─────────────────────────────────────
        level_div = self._check_level_divergence(
            market_price, sentiment, bayesian, echo
        )

        # ── 2. Velocity Divergence ──────────────────────────────────
        velocity_div = self._check_velocity_divergence(
            market_price, narrative, sentiment
        )

        # ── 3. Contrarian Divergence ────────────────────────────────
        contrarian_div = self._check_contrarian_divergence(
            market_price, echo, sentiment
        )

        # ── 4. Cross-Platform Divergence ────────────────────────────
        cross_div = self._check_cross_platform_divergence(
            market_price, sentiment
        )

        # ── Pick the strongest signal ───────────────────────────────
        candidates = [d for d in [level_div, velocity_div, contrarian_div, cross_div] if d]

        if not candidates:
            return None

        # Sort by edge * confidence (expected value)
        candidates.sort(key=lambda d: abs(d.edge_estimate) * d.confidence, reverse=True)
        best = candidates[0]

        # Fill in market info
        best.market_id = market_id
        best.question = question
        best.market_price = market_price

        # Final confidence adjustment
        best.confidence *= self._market_quality_multiplier(market)

        return best if abs(best.edge_estimate) >= config.DIVERGENCE_THRESHOLD else None

    def _check_level_divergence(
        self,
        market_price: float,
        sentiment: AggregatedSentiment,
        bayesian: BayesianEstimate,
        echo: EchoChamberReading,
    ) -> Optional[DivergenceSignal]:
        """
        When Bayesian posterior significantly differs from market price.
        """
        edge = bayesian.posterior - market_price

        if abs(edge) < config.DIVERGENCE_THRESHOLD:
            return None

        # Adjust for echo chamber
        adj_edge = edge * echo.confidence_adjustment
        if abs(adj_edge) < config.DIVERGENCE_THRESHOLD * 0.8:
            return None

        direction = "BUY_YES" if adj_edge > 0 else "BUY_NO"

        confidence = bayesian.confidence * sentiment.confidence
        confidence = min(confidence * 1.2, 0.95)  # Level divergence is our strongest signal

        strength = self._classify_strength(abs(adj_edge))

        return DivergenceSignal(
            market_id="",
            question="",
            signal_type="level",
            direction=direction,
            edge_estimate=float(adj_edge),
            confidence=float(confidence),
            market_price=market_price,
            sentiment_price=float(bayesian.posterior),
            bayesian_posterior=float(bayesian.posterior),
            strength=strength,
            explanation=(
                f"Level divergence: market={market_price:.1%}, "
                f"Bayesian={bayesian.posterior:.1%}, "
                f"edge={adj_edge:+.1%}. "
                f"Based on {sentiment.sample_size} data points. "
                f"{echo.explanation}"
            ),
        )

    def _check_velocity_divergence(
        self,
        market_price: float,
        narrative: NarrativeMomentum,
        sentiment: AggregatedSentiment,
    ) -> Optional[DivergenceSignal]:
        """
        When sentiment is rapidly shifting but price hasn't moved.
        Early narrative phases are most valuable.
        """
        velocity = narrative.velocity
        momentum = narrative.momentum_score

        if abs(momentum) < 0.2:
            return None  # No significant momentum

        # Estimate where sentiment is heading
        projected_sentiment = sentiment.yes_probability + velocity * 2
        projected_sentiment = float(np.clip(projected_sentiment, 0.05, 0.95))

        projected_edge = projected_sentiment - market_price

        if abs(projected_edge) < config.DIVERGENCE_THRESHOLD:
            return None

        direction = "BUY_YES" if projected_edge > 0 else "BUY_NO"

        # Confidence: higher for early/building narratives
        phase_conf = {
            "early": 0.8,
            "building": 0.7,
            "stable": 0.5,
            "peak": 0.3,
            "fading": 0.2,
        }.get(narrative.phase, 0.4)

        confidence = phase_conf * min(abs(momentum), 1.0)

        # Acceleration bonus
        if narrative.acceleration > 0 and velocity > 0:
            confidence *= 1.15
        elif narrative.acceleration < 0 and velocity < 0:
            confidence *= 1.15

        strength = self._classify_strength(abs(projected_edge))

        return DivergenceSignal(
            market_id="",
            question="",
            signal_type="velocity",
            direction=direction,
            edge_estimate=float(projected_edge),
            confidence=float(min(confidence, 0.85)),
            market_price=market_price,
            sentiment_price=float(projected_sentiment),
            bayesian_posterior=float(projected_sentiment),
            strength=strength,
            explanation=(
                f"Velocity divergence: sentiment moving at {velocity:+.3f}/window, "
                f"narrative phase={narrative.phase}. "
                f"Projected: {projected_sentiment:.1%} vs market {market_price:.1%}. "
                f"{narrative.explanation}"
            ),
        )

    def _check_contrarian_divergence(
        self,
        market_price: float,
        echo: EchoChamberReading,
        sentiment: AggregatedSentiment,
    ) -> Optional[DivergenceSignal]:
        """
        When the crowd is overwhelmingly in one direction AND the market
        disagrees, bet WITH the market AGAINST the crowd.

        This is the riskiest but often most profitable signal.
        Only triggered when echo chamber + significant price gap.
        """
        if not echo.contrarian_signal:
            return None

        # The crowd says one thing, the market says another
        crowd_direction = echo.dominant_direction
        crowd_pct = echo.dominant_pct

        if crowd_direction == "YES":
            # Crowd thinks YES, so contrarian = BUY_NO
            # Market is below crowd sentiment — trust market, fade crowd
            if market_price < sentiment.yes_probability - 0.10:
                raw_gap = sentiment.yes_probability - market_price
                edge = -(raw_gap * 0.5)  # Negative edge = BUY_NO
                direction = "BUY_NO"
            else:
                return None
        else:
            # Crowd thinks NO, so contrarian = BUY_YES
            # Market is above crowd sentiment — trust market, fade crowd
            if market_price > sentiment.yes_probability + 0.10:
                raw_gap = market_price - sentiment.yes_probability
                edge = raw_gap * 0.5  # Positive edge = BUY_YES
                direction = "BUY_YES"
            else:
                return None

        if abs(edge) < config.DIVERGENCE_THRESHOLD:
            return None

        confidence = 0.4  # Low base confidence for contrarian plays
        # Boost if source divergence supports it
        confidence *= (1 + echo.source_divergence)

        return DivergenceSignal(
            market_id="",
            question="",
            signal_type="contrarian",
            direction=direction,
            edge_estimate=float(edge),
            confidence=float(min(confidence, 0.65)),
            market_price=market_price,
            sentiment_price=float(sentiment.yes_probability),
            bayesian_posterior=float(market_price),  # Trust market here
            strength="moderate",
            explanation=(
                f"CONTRARIAN: {crowd_pct:.0%} of crowd says {crowd_direction}, "
                f"but market is at {market_price:.1%}. "
                f"Fading the echo chamber. {echo.explanation}"
            ),
        )

    def _check_cross_platform_divergence(
        self,
        market_price: float,
        sentiment: AggregatedSentiment,
    ) -> Optional[DivergenceSignal]:
        """
        When Reddit and Twitter disagree significantly.
        """
        breakdown = sentiment.source_breakdown
        if len(breakdown) < 2:
            return None

        reddit_yes = breakdown.get("reddit", 0.5)
        twitter_yes = breakdown.get("twitter", 0.5)

        platform_gap = abs(reddit_yes - twitter_yes)

        if platform_gap < 0.15:
            return None  # Platforms roughly agree

        # Which platform to trust? Heuristic:
        # - For politics: Twitter is faster, Reddit is more analytical
        # - For crypto: Reddit (r/cryptocurrency) is more informed
        # - Default: average, slightly favoring the one further from market
        reddit_diff = abs(reddit_yes - market_price)
        twitter_diff = abs(twitter_yes - market_price)

        # Trust the platform that disagrees MORE with the current price
        # (it might have information the market hasn't absorbed)
        if reddit_diff > twitter_diff:
            trusted = reddit_yes
            source = "Reddit"
        else:
            trusted = twitter_yes
            source = "Twitter"

        edge = trusted - market_price

        if abs(edge) < config.DIVERGENCE_THRESHOLD:
            return None

        direction = "BUY_YES" if edge > 0 else "BUY_NO"
        confidence = 0.45 * min(platform_gap * 3, 1.0)

        return DivergenceSignal(
            market_id="",
            question="",
            signal_type="cross_platform",
            direction=direction,
            edge_estimate=float(edge),
            confidence=float(min(confidence, 0.70)),
            market_price=market_price,
            sentiment_price=float(trusted),
            bayesian_posterior=float((reddit_yes + twitter_yes) / 2),
            strength=self._classify_strength(abs(edge)),
            explanation=(
                f"Cross-platform divergence: Reddit={reddit_yes:.1%}, "
                f"Twitter={twitter_yes:.1%} (gap={platform_gap:.1%}). "
                f"Trusting {source}. Edge={edge:+.1%}"
            ),
        )

    def _classify_strength(self, edge: float) -> str:
        if edge >= 0.20:
            return "strong"
        elif edge >= 0.12:
            return "moderate"
        return "weak"

    def _market_quality_multiplier(self, market: dict) -> float:
        """
        Adjust confidence based on market characteristics.
        Higher liquidity + volume = more reliable.
        """
        vol = market.get("volume_24h", 0)
        liq = market.get("liquidity", 0)

        # Volume multiplier
        if vol > 50000:
            vol_mult = 1.0
        elif vol > 10000:
            vol_mult = 0.85
        else:
            vol_mult = 0.65

        # Liquidity multiplier
        if liq > 100000:
            liq_mult = 1.0
        elif liq > 20000:
            liq_mult = 0.85
        else:
            liq_mult = 0.70

        return vol_mult * liq_mult
