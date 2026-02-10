"""
Bayesian Belief Updater
------------------------
The mathematical core of the prediction engine.

Approach:
  1. Start with the Polymarket price as the PRIOR probability
     (the market's current best estimate)
  2. Use sentiment data as EVIDENCE to update the probability
  3. Apply Bayes' theorem to get a POSTERIOR probability
  4. The DIFFERENCE between posterior and market price = our edge

This is powerful because:
  - We're not claiming to know the "true" probability
  - We're saying "given what Reddit/Twitter are saying, the market
    price is slightly off in this direction"
  - Even a 5% edge, repeatedly exploited, is hugely profitable

Uses Beta-Binomial conjugate model for clean updates.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

import config

logger = logging.getLogger(__name__)


@dataclass
class BayesianEstimate:
    """Result of Bayesian probability estimation."""
    prior: float                   # Market price (prior probability)
    posterior: float               # Our updated estimate
    edge: float                    # posterior - prior (our estimated edge)
    credible_interval: Tuple[float, float]  # 90% credible interval
    evidence_strength: float       # How much evidence moved the prior
    confidence: float              # How confident are we in the update
    n_observations: int
    explanation: str


class BayesianUpdater:
    """
    Bayesian belief updating using the market price as a prior
    and sentiment data as evidence.

    Uses a Beta distribution model:
      Prior:     Beta(alpha_prior, beta_prior) derived from market price
      Evidence:  Pseudocounts from sentiment analysis
      Posterior: Beta(alpha_post, beta_post) via conjugate update
    """

    def __init__(
        self,
        prior_weight: float = config.PRIOR_WEIGHT,
        evidence_weight: float = config.EVIDENCE_WEIGHT,
    ):
        self.prior_weight = prior_weight
        self.evidence_weight = evidence_weight

    def update(
        self,
        market_price: float,
        sentiment_yes_prob: float,
        sentiment_confidence: float,
        sample_size: int,
        echo_chamber_adj: float = 1.0,
    ) -> BayesianEstimate:
        """
        Update our belief about an event's probability.

        Args:
            market_price: Current Polymarket YES price (our prior)
            sentiment_yes_prob: Sentiment-implied YES probability
            sentiment_confidence: Confidence in the sentiment reading (0-1)
            sample_size: Number of sentiment data points
            echo_chamber_adj: Echo chamber confidence multiplier
        """
        # ── Construct Prior ─────────────────────────────────────────
        # Convert market price to Beta distribution parameters.
        # Higher prior_strength = more trust in the market price.
        prior_strength = self._compute_prior_strength(market_price)
        alpha_prior = market_price * prior_strength
        beta_prior = (1 - market_price) * prior_strength

        # ── Construct Evidence ──────────────────────────────────────
        # Convert sentiment into pseudocounts.
        # More confident + more data = more pseudocounts = more influence.
        evidence_strength = self._compute_evidence_strength(
            sentiment_confidence, sample_size, echo_chamber_adj
        )

        alpha_evidence = sentiment_yes_prob * evidence_strength
        beta_evidence = (1 - sentiment_yes_prob) * evidence_strength

        # ── Bayesian Update (conjugate) ─────────────────────────────
        alpha_post = alpha_prior + alpha_evidence
        beta_post = beta_prior + beta_evidence

        # Posterior mean
        posterior = alpha_post / (alpha_post + beta_post)

        # Clip to bounds
        posterior = float(np.clip(
            posterior,
            config.BAYESIAN_CONFIDENCE_FLOOR,
            config.BAYESIAN_CONFIDENCE_CEILING,
        ))

        # ── Credible Interval ───────────────────────────────────────
        # 90% highest density interval
        try:
            dist = stats.beta(alpha_post, beta_post)
            ci_low = float(dist.ppf(0.05))
            ci_high = float(dist.ppf(0.95))
        except Exception:
            ci_low = max(posterior - 0.15, 0.01)
            ci_high = min(posterior + 0.15, 0.99)

        # ── Edge Calculation ────────────────────────────────────────
        edge = posterior - market_price

        # ── Confidence in the Update ────────────────────────────────
        # High confidence when: large sample, strong agreement,
        # no echo chamber, and the update isn't suspiciously large
        update_size = abs(edge)
        size_penalty = 1.0 if update_size < 0.15 else max(0.5, 1.0 - update_size)

        confidence = (
            min(sentiment_confidence * echo_chamber_adj, 1.0)
            * min(sample_size / 30, 1.0)     # More data = more confident
            * size_penalty                     # Huge updates are suspicious
        )
        confidence = float(np.clip(confidence, 0.05, 0.95))

        explanation = (
            f"Prior: {market_price:.1%} (market) | "
            f"Evidence: {sentiment_yes_prob:.1%} (sentiment, n={sample_size}) | "
            f"Posterior: {posterior:.1%} | "
            f"Edge: {edge:+.1%} | "
            f"90% CI: [{ci_low:.1%}, {ci_high:.1%}]"
        )

        return BayesianEstimate(
            prior=market_price,
            posterior=posterior,
            edge=edge,
            credible_interval=(ci_low, ci_high),
            evidence_strength=evidence_strength,
            confidence=confidence,
            n_observations=sample_size,
            explanation=explanation,
        )

    def _compute_prior_strength(self, market_price: float) -> float:
        """
        How much to trust the market price.

        Markets near 50/50 are harder to call — less prior conviction.
        Markets near extremes (95% YES) have strong prior conviction.
        """
        # Distance from 50%
        extremity = abs(market_price - 0.5) * 2  # 0 at 50%, 1 at extremes

        # Base strength (how many pseudo-observations the prior is worth)
        base = 20  # Market price "feels like" 20 data points

        # More extreme prices = stronger prior (the market is more sure)
        strength = base * (1 + extremity * 2)

        return strength * self.prior_weight

    def _compute_evidence_strength(
        self,
        confidence: float,
        sample_size: int,
        echo_chamber_adj: float,
    ) -> float:
        """
        How many pseudocounts the sentiment evidence is worth.
        More confident analysis + more data = more influence.
        """
        # Base evidence: each data point adds proportional influence
        # But with diminishing returns (sqrt)
        base = np.sqrt(max(sample_size, 1)) * 2

        # Scale by confidence and echo chamber adjustment
        strength = base * confidence * echo_chamber_adj

        # Cap evidence so it can't overwhelm the prior too much
        max_evidence = 30
        strength = min(strength, max_evidence)

        return strength * self.evidence_weight

    def multi_update(
        self,
        market_price: float,
        evidence_list: List[dict],
    ) -> BayesianEstimate:
        """
        Apply multiple evidence sources sequentially.

        Each item in evidence_list should have:
          - yes_prob: float
          - confidence: float
          - sample_size: int
          - echo_adj: float (optional, default 1.0)
        """
        current_price = market_price

        for ev in evidence_list:
            result = self.update(
                market_price=current_price,
                sentiment_yes_prob=ev["yes_prob"],
                sentiment_confidence=ev["confidence"],
                sample_size=ev["sample_size"],
                echo_chamber_adj=ev.get("echo_adj", 1.0),
            )
            current_price = result.posterior

        # Final result with original market price as prior
        return BayesianEstimate(
            prior=market_price,
            posterior=current_price,
            edge=current_price - market_price,
            credible_interval=result.credible_interval,
            evidence_strength=result.evidence_strength,
            confidence=result.confidence,
            n_observations=sum(e["sample_size"] for e in evidence_list),
            explanation=f"Multi-source update: {market_price:.1%} -> {current_price:.1%}",
        )
