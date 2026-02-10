"""
Advanced Sentiment Analyzer
-----------------------------
Multi-layer NLP pipeline that scores text for directional sentiment
(does this text suggest YES or NO for a given Polymarket question?).

Pipeline:
  1. VADER — fast, rule-based sentiment (good for social media)
  2. TextBlob — pattern-based polarity & subjectivity
  3. Keyword scoring — domain-specific boosted terms
  4. Directional mapping — maps general sentiment to YES/NO probability
  5. Time-decay weighting — recent posts count more
  6. Credibility weighting — high-karma/verified authors count more

The output is a float from 0.0 (strong NO) to 1.0 (strong YES) for
each market question.
"""

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# Lazy imports — these can be heavy
_vader = None
_textblob = None


def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed")
    return _vader


def _get_textblob():
    global _textblob
    if _textblob is None:
        try:
            import textblob as tb
            _textblob = tb
        except ImportError:
            logger.warning("textblob not installed")
    return _textblob


# ── Keyword dictionaries for prediction markets ────────────────────────

# Words that suggest YES outcome
YES_KEYWORDS = {
    # High confidence
    "confirmed": 2.0, "definitely": 1.8, "guaranteed": 1.8,
    "certain": 1.7, "inevitable": 1.7, "absolutely": 1.6,
    "locked in": 1.5, "done deal": 1.5, "for sure": 1.5,
    # Medium confidence
    "likely": 1.2, "probably": 1.2, "expected": 1.1,
    "favored": 1.1, "winning": 1.1, "ahead": 1.0,
    "leading": 1.0, "strong": 0.9, "bullish": 0.9,
    "positive": 0.8, "agree": 0.8, "yes": 0.8,
    "support": 0.7, "momentum": 0.7, "rising": 0.7,
    "surge": 0.8, "rally": 0.8, "breakout": 0.7,
}

# Words that suggest NO outcome
NO_KEYWORDS = {
    # High confidence
    "impossible": 2.0, "no way": 1.8, "never": 1.7,
    "zero chance": 1.8, "debunked": 1.7, "denied": 1.5,
    "rejected": 1.5, "failed": 1.4, "collapsed": 1.4,
    # Medium confidence
    "unlikely": 1.2, "doubtful": 1.2, "improbable": 1.1,
    "struggling": 1.0, "losing": 1.0, "behind": 0.9,
    "weak": 0.9, "bearish": 0.9, "negative": 0.8,
    "disagree": 0.8, "no": 0.7, "against": 0.7,
    "crash": 0.8, "dump": 0.8, "decline": 0.7,
    "slump": 0.7, "drop": 0.7,
}

# Hedging / uncertainty modifiers (reduce confidence)
HEDGE_WORDS = {
    "maybe", "might", "could", "possibly", "perhaps",
    "uncertain", "unclear", "idk", "not sure", "hard to say",
    "50/50", "coin flip", "toss up", "who knows",
}


@dataclass
class SentimentResult:
    """Result of analyzing one text against a market question."""
    yes_probability: float     # 0.0 to 1.0
    confidence: float          # 0.0 to 1.0 (how sure are we about the reading)
    vader_compound: float      # Raw VADER score (-1 to 1)
    textblob_polarity: float   # Raw TextBlob polarity (-1 to 1)
    textblob_subjectivity: float
    keyword_score: float       # Domain keyword score
    directional: str           # "YES", "NO", "NEUTRAL"


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across all data points for a market."""
    market_id: str
    question: str
    yes_probability: float            # Weighted average YES prob
    confidence: float                 # Overall confidence
    sample_size: int                  # Number of data points
    sentiment_std: float              # Standard deviation (disagreement)
    source_breakdown: Dict[str, float]  # Per-source YES probability
    time_series: List[Tuple[datetime, float]]  # Sentiment over time
    bullish_pct: float                # % of texts leaning YES
    bearish_pct: float                # % of texts leaning NO
    avg_credibility: float            # Average source credibility


class SentimentAnalyzer:
    """
    Analyzes text data points to estimate YES/NO probability for
    Polymarket questions.
    """

    def analyze_text(
        self,
        text: str,
        market_question: str,
        direction_context: str = "",
    ) -> SentimentResult:
        """
        Analyze a single text against a market question.

        Args:
            text: The content to analyze
            market_question: The Polymarket YES/NO question
            direction_context: Optional context like "Trump wins" to help
                               map sentiment direction correctly
        """
        if not text or len(text.strip()) < 5:
            return SentimentResult(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, "NEUTRAL")

        text_lower = text.lower()

        # ── Layer 1: VADER ──────────────────────────────────────────
        vader_score = self._vader_sentiment(text)

        # ── Layer 2: TextBlob ───────────────────────────────────────
        tb_polarity, tb_subjectivity = self._textblob_sentiment(text)

        # ── Layer 3: Keyword scoring ────────────────────────────────
        keyword_score = self._keyword_sentiment(text_lower)

        # ── Layer 4: Directional mapping ────────────────────────────
        # Map general sentiment to YES/NO for this specific question
        yes_prob = self._map_to_direction(
            vader_score, tb_polarity, keyword_score,
            text_lower, market_question.lower(), direction_context.lower(),
        )

        # ── Layer 5: Confidence estimation ──────────────────────────
        confidence = self._estimate_confidence(
            vader_score, tb_polarity, tb_subjectivity,
            keyword_score, text_lower,
        )

        directional = "YES" if yes_prob > 0.55 else ("NO" if yes_prob < 0.45 else "NEUTRAL")

        return SentimentResult(
            yes_probability=yes_prob,
            confidence=confidence,
            vader_compound=vader_score,
            textblob_polarity=tb_polarity,
            textblob_subjectivity=tb_subjectivity,
            keyword_score=keyword_score,
            directional=directional,
        )

    def analyze_batch(
        self,
        data_points: list,  # List[SentimentDataPoint]
        market_id: str,
        market_question: str,
    ) -> AggregatedSentiment:
        """
        Analyze all scraped data points for a market and produce
        a single aggregated sentiment reading.
        """
        if not data_points:
            return AggregatedSentiment(
                market_id=market_id, question=market_question,
                yes_probability=0.5, confidence=0.0, sample_size=0,
                sentiment_std=0.0, source_breakdown={},
                time_series=[], bullish_pct=0.0, bearish_pct=0.0,
                avg_credibility=0.0,
            )

        results = []
        weights = []
        time_series = []
        source_yes = {}
        source_counts = {}

        now = datetime.now(timezone.utc)

        for dp in data_points:
            result = self.analyze_text(dp.text, market_question)

            # Time decay weight
            age_hours = max(
                (now - dp.timestamp).total_seconds() / 3600, 0.1
            )
            time_weight = math.exp(
                -0.693 * age_hours / config.SENTIMENT_HALF_LIFE_HOURS
            )

            # Combined weight: credibility * time_decay * confidence
            w = dp.credibility_weight * time_weight * max(result.confidence, 0.1)

            results.append(result)
            weights.append(w)
            time_series.append((dp.timestamp, result.yes_probability))

            # Source breakdown
            src = dp.source
            source_yes.setdefault(src, 0.0)
            source_counts.setdefault(src, 0.0)
            source_yes[src] += result.yes_probability * w
            source_counts[src] += w

        # Weighted average YES probability
        total_weight = sum(weights)
        if total_weight > 0:
            yes_probs = [r.yes_probability for r in results]
            weighted_yes = sum(p * w for p, w in zip(yes_probs, weights)) / total_weight
        else:
            weighted_yes = 0.5

        # Standard deviation (measures disagreement)
        yes_probs_arr = np.array([r.yes_probability for r in results])
        sentiment_std = float(np.std(yes_probs_arr)) if len(yes_probs_arr) > 1 else 0.0

        # Directional counts
        bullish = sum(1 for r in results if r.yes_probability > 0.55)
        bearish = sum(1 for r in results if r.yes_probability < 0.45)
        n = len(results)

        # Source breakdown averages
        breakdown = {}
        for src in source_yes:
            if source_counts[src] > 0:
                breakdown[src] = source_yes[src] / source_counts[src]

        # Overall confidence
        avg_conf = np.mean([r.confidence for r in results])
        # Boost confidence with more data
        sample_bonus = min(len(results) / 50, 0.3)
        overall_conf = min(float(avg_conf) + sample_bonus, 1.0)

        avg_cred = np.mean([dp.credibility_weight for dp in data_points])

        # Sort time series
        time_series.sort(key=lambda x: x[0])

        return AggregatedSentiment(
            market_id=market_id,
            question=market_question,
            yes_probability=float(np.clip(weighted_yes, 0.01, 0.99)),
            confidence=float(overall_conf),
            sample_size=n,
            sentiment_std=float(sentiment_std),
            source_breakdown=breakdown,
            time_series=time_series,
            bullish_pct=bullish / max(n, 1),
            bearish_pct=bearish / max(n, 1),
            avg_credibility=float(avg_cred),
        )

    # ── Internal Methods ────────────────────────────────────────────────

    def _vader_sentiment(self, text: str) -> float:
        """VADER compound score: -1 (negative) to +1 (positive)."""
        analyzer = _get_vader()
        if analyzer is None:
            return 0.0
        scores = analyzer.polarity_scores(text)
        return scores["compound"]

    def _textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """TextBlob polarity (-1 to 1) and subjectivity (0 to 1)."""
        tb = _get_textblob()
        if tb is None:
            return 0.0, 0.5
        blob = tb.TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def _keyword_sentiment(self, text_lower: str) -> float:
        """
        Score text using prediction-market-specific keywords.
        Returns: negative = NO leaning, positive = YES leaning.
        """
        yes_score = 0.0
        no_score = 0.0

        for keyword, weight in YES_KEYWORDS.items():
            if keyword in text_lower:
                yes_score += weight

        for keyword, weight in NO_KEYWORDS.items():
            if keyword in text_lower:
                no_score += weight

        # Hedge words reduce the magnitude
        hedge_count = sum(1 for hw in HEDGE_WORDS if hw in text_lower)
        if hedge_count > 0:
            dampener = 0.5 ** hedge_count
            yes_score *= dampener
            no_score *= dampener

        total = yes_score + no_score
        if total == 0:
            return 0.0
        return (yes_score - no_score) / max(total, 1)

    def _map_to_direction(
        self,
        vader: float,
        textblob: float,
        keyword: float,
        text: str,
        question: str,
        context: str,
    ) -> float:
        """
        Map general sentiment scores to YES/NO probability for the question.

        Key insight: positive sentiment about the SUBJECT of a YES question
        maps to higher YES probability. But "this will never happen" is
        negative sentiment that maps to NO.
        """
        # Weighted combination of raw scores
        w = config.SENTIMENT_WEIGHTS
        raw = (
            vader * w["vader"]
            + textblob * w["textblob"]
            + keyword * w["keyword"]
        )

        # Check for explicit YES/NO statements in the text
        explicit_yes = any(p in text for p in [
            "yes", "will happen", "going to happen", "bet yes",
            "buy yes", "i think so", "100%", "slam dunk",
        ])
        explicit_no = any(p in text for p in [
            "no way", "won't happen", "not going to", "bet no",
            "buy no", "i don't think", "0%", "never going to",
        ])

        if explicit_yes and not explicit_no:
            raw = max(raw, 0.3)
        elif explicit_no and not explicit_yes:
            raw = min(raw, -0.3)

        # Check for negation patterns near the question subject
        negation_near = bool(re.search(
            r"\b(not|no|never|won't|can't|couldn't|shouldn't|unlikely)\b.*"
            r"\b(happen|win|pass|succeed|reach|hit)\b",
            text,
        ))
        if negation_near:
            raw = min(raw, -abs(raw))

        # Map [-1, 1] to [0, 1] probability
        yes_prob = 0.5 + (raw * 0.5)
        return float(np.clip(yes_prob, 0.02, 0.98))

    def _estimate_confidence(
        self,
        vader: float,
        textblob: float,
        subjectivity: float,
        keyword: float,
        text: str,
    ) -> float:
        """
        Estimate how confident we are in this sentiment reading.
        Low confidence when: sources disagree, text is vague, hedging.
        """
        # Agreement between methods boosts confidence
        scores = [vader, textblob, keyword]
        signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in scores]

        if all(s == signs[0] for s in signs) and signs[0] != 0:
            agreement = 0.8  # All agree
        elif sum(s != 0 for s in signs) <= 1:
            agreement = 0.3  # Only one has an opinion
        else:
            agreement = 0.5

        # Magnitude: stronger signals = higher confidence
        magnitude = np.mean([abs(s) for s in scores])
        mag_conf = min(magnitude * 1.5, 0.8)

        # Subjectivity penalty: highly subjective text is less reliable
        subj_penalty = max(1 - subjectivity * 0.3, 0.5)

        # Text length bonus: longer = usually more informative
        length_bonus = min(len(text) / 500, 0.2)

        # Hedge penalty
        hedge_count = sum(1 for hw in HEDGE_WORDS if hw in text)
        hedge_penalty = max(1 - hedge_count * 0.15, 0.3)

        confidence = (agreement * 0.4 + mag_conf * 0.3 + length_bonus) * subj_penalty * hedge_penalty
        return float(np.clip(confidence, 0.05, 0.95))
