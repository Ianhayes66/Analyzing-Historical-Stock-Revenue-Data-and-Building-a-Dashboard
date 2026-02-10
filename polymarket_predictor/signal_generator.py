"""
Signal Generator
-----------------
The orchestrator that combines all components into final trading signals.

Pipeline:
  1. Polymarket Client   → Fetch active markets
  2. Reddit + Twitter    → Scrape sentiment data
  3. Sentiment Analyzer  → NLP scoring
  4. Echo Chamber        → Detect uniformity / contrarian opportunities
  5. Narrative Tracker   → Measure sentiment velocity & phase
  6. Bayesian Updater    → Update probability estimate
  7. Divergence Engine   → Find price-sentiment gaps
  8. Bankroll Manager    → Size the bet

Each step feeds the next. The final output is a ranked list of
actionable opportunities with bet sizes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
from polymarket_predictor.bankroll_manager import BankrollManager
from polymarket_predictor.bayesian_updater import BayesianUpdater
from polymarket_predictor.divergence_engine import DivergenceEngine, DivergenceSignal
from polymarket_predictor.echo_chamber import (
    EchoChamberDetector,
    EchoChamberReading,
    NarrativeMomentum,
    NarrativeTracker,
)
from polymarket_predictor.polymarket_client import (
    fetch_active_markets,
    filter_markets,
)
from polymarket_predictor.reddit_scraper import RedditScraper
from polymarket_predictor.sentiment_analyzer import AggregatedSentiment, SentimentAnalyzer
from polymarket_predictor.twitter_scraper import TwitterScraper

logger = logging.getLogger(__name__)


@dataclass
class Opportunity:
    """A fully-scored trading opportunity."""
    market: dict
    signal: DivergenceSignal
    sentiment: AggregatedSentiment
    echo: EchoChamberReading
    narrative: NarrativeMomentum
    recommended_bet: float            # Dollar amount (0 = don't bet)
    expected_value: float             # Edge * bet size
    composite_score: float            # Final ranking score (0-1)


class SignalGenerator:
    """
    Master orchestrator: turns raw data into ranked trading opportunities.
    """

    def __init__(self, bankroll_mgr: Optional[BankrollManager] = None):
        self.reddit = RedditScraper()
        self.twitter = TwitterScraper()
        self.sentiment = SentimentAnalyzer()
        self.echo_detector = EchoChamberDetector()
        self.narrative_tracker = NarrativeTracker()
        self.bayesian = BayesianUpdater()
        self.divergence = DivergenceEngine()
        self.bankroll = bankroll_mgr or BankrollManager()

        self._last_scan: Optional[datetime] = None
        self._opportunities: List[Opportunity] = []

    def scan(self) -> List[Opportunity]:
        """
        Full scan: fetch markets → analyze sentiment → generate signals.
        Returns opportunities sorted by composite score (best first).
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL MARKET SCAN")
        logger.info("=" * 60)

        # Step 1: Fetch active markets
        raw_markets = fetch_active_markets(limit=100)
        markets = filter_markets(raw_markets)
        logger.info(f"Found {len(markets)} tradeable markets")

        opportunities = []

        for i, market in enumerate(markets):
            try:
                logger.info(
                    f"[{i+1}/{len(markets)}] Analyzing: "
                    f"{market['question'][:70]}..."
                )
                opp = self._analyze_market(market)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.error(
                    f"Error analyzing market '{market.get('question', '?')[:50]}': {e}",
                    exc_info=True,
                )

        # Sort by composite score
        opportunities.sort(key=lambda o: o.composite_score, reverse=True)

        self._opportunities = opportunities
        self._last_scan = datetime.now(timezone.utc)

        logger.info(f"Scan complete: {len(opportunities)} opportunities found")
        return opportunities

    def get_top_opportunities(self, n: int = 10) -> List[Opportunity]:
        """Get the top N opportunities from the last scan."""
        return self._opportunities[:n]

    def _analyze_market(self, market: dict) -> Optional[Opportunity]:
        """Full analysis pipeline for a single market."""
        question = market["question"]
        market_id = market["id"]
        market_price = market["yes_price"]

        # ── Step 2: Scrape sentiment ────────────────────────────────
        reddit_data = self.reddit.search_market(question)
        twitter_data = self.twitter.search_market(question)
        all_data = reddit_data + twitter_data

        if len(all_data) < 5:
            logger.debug(f"Skipping '{question[:50]}' — only {len(all_data)} data points")
            return None

        # ── Step 3: Analyze sentiment ───────────────────────────────
        agg_sentiment = self.sentiment.analyze_batch(
            all_data, market_id, question
        )

        if agg_sentiment.sample_size < config.MIN_SAMPLE_SIZE:
            return None

        # ── Step 4: Echo chamber detection ──────────────────────────
        echo = self.echo_detector.analyze(
            sentiment_data=[],  # Individual results not stored, using aggregates
            source_breakdown=agg_sentiment.source_breakdown,
            bullish_pct=agg_sentiment.bullish_pct,
            bearish_pct=agg_sentiment.bearish_pct,
            sample_size=agg_sentiment.sample_size,
        )

        # ── Step 5: Narrative momentum ──────────────────────────────
        narrative = self.narrative_tracker.analyze(
            agg_sentiment.time_series,
            agg_sentiment.yes_probability,
        )

        # ── Step 6: Bayesian update ─────────────────────────────────
        bayesian = self.bayesian.update(
            market_price=market_price,
            sentiment_yes_prob=agg_sentiment.yes_probability,
            sentiment_confidence=agg_sentiment.confidence,
            sample_size=agg_sentiment.sample_size,
            echo_chamber_adj=echo.confidence_adjustment,
        )

        # ── Step 7: Divergence detection ────────────────────────────
        signal = self.divergence.analyze(
            market, agg_sentiment, bayesian, echo, narrative
        )

        if signal is None:
            return None

        # ── Step 8: Bet sizing ──────────────────────────────────────
        bet_direction = "YES" if signal.direction == "BUY_YES" else "NO"
        bet_price = market_price if bet_direction == "YES" else (1 - market_price)

        bet_size = self.bankroll.calculate_bet_size(
            estimated_prob=bayesian.posterior if bet_direction == "YES" else (1 - bayesian.posterior),
            market_price=bet_price,
            direction=bet_direction,
            confidence=signal.confidence,
        )

        # ── Composite Score ─────────────────────────────────────────
        composite = self._compute_composite_score(
            signal, agg_sentiment, echo, narrative, bayesian
        )

        ev = bet_size * abs(signal.edge_estimate) if bet_size > 0 else 0

        opp = Opportunity(
            market=market,
            signal=signal,
            sentiment=agg_sentiment,
            echo=echo,
            narrative=narrative,
            recommended_bet=bet_size,
            expected_value=round(ev, 4),
            composite_score=composite,
        )

        logger.info(
            f"  → OPPORTUNITY: {signal.direction} | "
            f"Edge: {signal.edge_estimate:+.1%} | "
            f"Confidence: {signal.confidence:.0%} | "
            f"Bet: ${bet_size:.2f} | "
            f"Score: {composite:.2f}"
        )

        return opp

    def _compute_composite_score(
        self,
        signal: DivergenceSignal,
        sentiment: AggregatedSentiment,
        echo: EchoChamberReading,
        narrative: NarrativeMomentum,
        bayesian,
    ) -> float:
        """
        Compute a final composite score for ranking opportunities.
        Uses the configured component weights.
        """
        w = config.SIGNAL_COMPONENTS

        # Divergence component (primary)
        div_score = min(abs(signal.edge_estimate) * 5, 1.0) * signal.confidence

        # Velocity component
        vel_score = min(abs(narrative.momentum_score), 1.0)
        # Boost if velocity direction matches signal direction
        if (narrative.velocity > 0 and signal.direction == "BUY_YES") or \
           (narrative.velocity < 0 and signal.direction == "BUY_NO"):
            vel_score *= 1.3

        # Echo chamber component (inverted — diversity is good)
        echo_score = 1.0 - echo.uniformity_score
        if echo.contrarian_signal:
            echo_score = 0.8  # Contrarian signals score well

        # Bayesian confidence
        bayes_score = bayesian.confidence * min(abs(bayesian.edge) * 5, 1.0)

        # Volume-sentiment alignment
        vol_sent_score = 0.5
        if sentiment.sample_size > 30:
            vol_sent_score = min(sentiment.confidence * 1.2, 1.0)

        composite = (
            div_score * w["divergence"]
            + vel_score * w["velocity"]
            + echo_score * w["echo_chamber"]
            + bayes_score * w["bayesian"]
            + vol_sent_score * w["volume_sentiment"]
        )

        return float(min(composite, 1.0))

    def print_opportunities(self, opportunities: Optional[List[Opportunity]] = None):
        """Print a formatted table of opportunities."""
        opps = opportunities or self._opportunities

        if not opps:
            print("\nNo opportunities found in latest scan.")
            return

        print("\n" + "=" * 110)
        print(f"{'TOP OPPORTUNITIES':^110}")
        print("=" * 110)
        print(
            f"{'#':<4} {'Score':>6} {'Direction':>10} {'Edge':>7} "
            f"{'Conf':>6} {'Bet$':>7} {'EV$':>7} {'Type':<14} "
            f"{'Market Price':>12} {'Question':<30}"
        )
        print("-" * 110)

        for i, opp in enumerate(opps[:20], 1):
            q = opp.market["question"][:28]
            print(
                f"{i:<4} {opp.composite_score:>5.2f} "
                f"{opp.signal.direction:>10} "
                f"{opp.signal.edge_estimate:>+6.1%} "
                f"{opp.signal.confidence:>5.0%} "
                f"${opp.recommended_bet:>6.2f} "
                f"${opp.expected_value:>6.2f} "
                f"{opp.signal.signal_type:<14} "
                f"{opp.signal.market_price:>11.1%} "
                f"{q}"
            )

        print("=" * 110)
        bankroll = self.bankroll.get_summary()
        print(
            f"\nBankroll: ${bankroll['bankroll']:.2f} | "
            f"Cash: ${bankroll['cash']:.2f} | "
            f"Open Bets: {bankroll['open_bets']} | "
            f"P&L: ${bankroll['total_pnl']:.2f} | "
            f"ROI: {bankroll['roi_pct']:.1f}%"
        )
