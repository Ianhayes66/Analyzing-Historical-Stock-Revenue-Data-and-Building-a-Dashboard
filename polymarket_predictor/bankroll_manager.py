"""
Kelly Criterion Bankroll Manager
----------------------------------
Mathematically optimal bet sizing for the $50 bankroll.

Kelly Criterion formula:
    f* = (bp - q) / b
where:
    b = odds received (net odds: payout/bet - 1)
    p = probability of winning (our Bayesian estimate)
    q = probability of losing (1 - p)
    f* = fraction of bankroll to bet

We use QUARTER-KELLY (f*/4) because:
  1. Full Kelly is extremely aggressive and assumes perfect probability
  2. Our probabilities are estimates, not certain
  3. Quarter-Kelly reduces variance by 93% with only 25% less growth
  4. With $50, we can't afford a drawdown

Additional rules:
  - Never bet more than 10% of bankroll on one market ($5 max)
  - Never bet if estimated edge < 8%
  - Max 5 concurrent bets
  - Track everything for performance analysis
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class Bet:
    """A single bet placed on a Polymarket market."""
    market_id: str
    question: str
    direction: str              # "YES" or "NO"
    amount: float               # Dollar amount
    entry_price: float          # Price paid per share
    shares: float               # Number of shares
    estimated_edge: float       # Our estimated edge at time of bet
    confidence: float
    signal_type: str            # "level", "velocity", "contrarian", "cross_platform"
    timestamp: str
    status: str = "open"        # "open", "won", "lost", "sold"
    exit_price: float = 0.0
    pnl: float = 0.0


@dataclass
class BankrollState:
    """Full state of the bankroll."""
    cash: float = config.BANKROLL
    total_deposited: float = config.BANKROLL
    bets: List[Bet] = field(default_factory=list)
    closed_bets: List[Bet] = field(default_factory=list)

    @property
    def open_bets(self) -> List[Bet]:
        return [b for b in self.bets if b.status == "open"]

    @property
    def invested(self) -> float:
        return sum(b.amount for b in self.open_bets)

    @property
    def total_value(self) -> float:
        return self.cash + self.invested

    @property
    def total_pnl(self) -> float:
        return sum(b.pnl for b in self.closed_bets)

    @property
    def win_rate(self) -> float:
        wins = [b for b in self.closed_bets if b.pnl > 0]
        total = len(self.closed_bets)
        return len(wins) / max(total, 1)

    @property
    def roi_pct(self) -> float:
        if self.total_deposited == 0:
            return 0.0
        return (self.total_value - self.total_deposited) / self.total_deposited * 100


class BankrollManager:
    """
    Manages the $50 bankroll with Kelly Criterion sizing.
    Persists state to disk so you don't lose tracking between sessions.
    """

    def __init__(self, state_file: str = config.BANKROLL_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def calculate_bet_size(
        self,
        estimated_prob: float,
        market_price: float,
        direction: str,
        confidence: float,
    ) -> float:
        """
        Calculate optimal bet size using Quarter-Kelly Criterion.

        Args:
            estimated_prob: Our estimated probability (Bayesian posterior)
            market_price: Current Polymarket price (what we'd pay)
            direction: "YES" or "NO"
            confidence: Signal confidence (0-1)

        Returns:
            Bet size in dollars (0 if we shouldn't bet)
        """
        bankroll = self.state.total_value

        if bankroll < config.MIN_BET_SIZE:
            logger.warning("Bankroll depleted. No more bets.")
            return 0.0

        # Determine our edge and the effective odds
        if direction == "YES":
            our_prob = estimated_prob
            cost = market_price  # We pay this per share
            payout = 1.0         # Share pays $1 if YES
        else:
            our_prob = 1.0 - estimated_prob
            cost = 1.0 - market_price
            payout = 1.0

        # Net odds: what we receive per dollar risked
        if cost <= 0 or cost >= 1:
            return 0.0
        b = (payout / cost) - 1  # Net odds

        q = 1.0 - our_prob
        edge = b * our_prob - q

        # Must have minimum edge
        if edge < config.MIN_EDGE_TO_BET:
            logger.debug(
                f"Edge too small ({edge:.1%} < {config.MIN_EDGE_TO_BET:.1%}). "
                f"No bet."
            )
            return 0.0

        # Kelly fraction
        kelly = (b * our_prob - q) / b if b > 0 else 0.0
        kelly = max(kelly, 0.0)

        # Quarter-Kelly, adjusted by confidence
        fraction = kelly * config.KELLY_FRACTION * confidence

        # Absolute bet size
        bet_size = bankroll * fraction

        # Apply limits
        max_bet = bankroll * config.MAX_BET_PCT
        bet_size = min(bet_size, max_bet)
        bet_size = max(bet_size, 0.0)

        # Check cash availability
        bet_size = min(bet_size, self.state.cash)

        # Must meet minimum
        if bet_size < config.MIN_BET_SIZE:
            return 0.0

        # Round to 2 decimals
        bet_size = round(bet_size, 2)

        logger.info(
            f"Kelly sizing: edge={edge:.1%}, kelly={kelly:.1%}, "
            f"quarter={fraction:.1%}, bet=${bet_size:.2f} "
            f"(of ${bankroll:.2f} bankroll)"
        )

        return bet_size

    def can_place_bet(self, market_id: str) -> tuple:
        """Check if we can place a new bet."""
        if len(self.state.open_bets) >= config.MAX_CONCURRENT_BETS:
            return False, f"Max concurrent bets reached ({config.MAX_CONCURRENT_BETS})"

        if self.state.cash < config.MIN_BET_SIZE:
            return False, f"Insufficient cash (${self.state.cash:.2f})"

        # Don't bet on the same market twice
        for bet in self.state.open_bets:
            if bet.market_id == market_id:
                return False, f"Already have a bet on this market"

        return True, "OK"

    def place_bet(
        self,
        market_id: str,
        question: str,
        direction: str,
        amount: float,
        price: float,
        edge: float,
        confidence: float,
        signal_type: str,
    ) -> Optional[Bet]:
        """Place a bet and update bankroll state."""
        can_bet, reason = self.can_place_bet(market_id)
        if not can_bet:
            logger.warning(f"Cannot place bet: {reason}")
            return None

        if amount < config.MIN_BET_SIZE:
            return None

        shares = amount / price
        bet = Bet(
            market_id=market_id,
            question=question[:200],
            direction=direction,
            amount=round(amount, 2),
            entry_price=round(price, 4),
            shares=round(shares, 4),
            estimated_edge=round(edge, 4),
            confidence=round(confidence, 4),
            signal_type=signal_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.state.cash -= amount
        self.state.bets.append(bet)
        self._save_state()

        logger.info(
            f"BET PLACED: {direction} on '{question[:60]}' | "
            f"${amount:.2f} @ {price:.2f} ({shares:.1f} shares) | "
            f"Edge: {edge:.1%} | Cash remaining: ${self.state.cash:.2f}"
        )
        return bet

    def close_bet(
        self, market_id: str, exit_price: float, status: str = "sold"
    ) -> Optional[Bet]:
        """Close a bet and record P&L."""
        for bet in self.state.bets:
            if bet.market_id == market_id and bet.status == "open":
                bet.status = status
                bet.exit_price = exit_price

                if bet.direction == "YES":
                    if status == "won":
                        bet.pnl = bet.shares * 1.0 - bet.amount
                    elif status == "lost":
                        bet.pnl = -bet.amount
                    else:
                        bet.pnl = bet.shares * exit_price - bet.amount
                else:
                    if status == "won":
                        bet.pnl = bet.shares * 1.0 - bet.amount
                    elif status == "lost":
                        bet.pnl = -bet.amount
                    else:
                        bet.pnl = bet.shares * (1.0 - exit_price) - bet.amount

                self.state.cash += bet.amount + bet.pnl
                self.state.closed_bets.append(bet)
                self.state.bets = [b for b in self.state.bets if b.market_id != market_id or b.status != status]

                self._save_state()

                symbol = "+" if bet.pnl >= 0 else ""
                logger.info(
                    f"BET CLOSED: {bet.direction} '{bet.question[:60]}' | "
                    f"{status.upper()} | PnL: {symbol}${bet.pnl:.2f} | "
                    f"Cash: ${self.state.cash:.2f}"
                )
                return bet

        return None

    def get_summary(self) -> dict:
        """Get bankroll summary for dashboard."""
        return {
            "bankroll": round(self.state.total_value, 2),
            "cash": round(self.state.cash, 2),
            "invested": round(self.state.invested, 2),
            "total_pnl": round(self.state.total_pnl, 2),
            "roi_pct": round(self.state.roi_pct, 2),
            "open_bets": len(self.state.open_bets),
            "closed_bets": len(self.state.closed_bets),
            "win_rate": round(self.state.win_rate * 100, 1),
            "avg_bet_size": round(
                np.mean([b.amount for b in self.state.closed_bets]) if self.state.closed_bets else 0, 2
            ),
            "best_trade": round(
                max((b.pnl for b in self.state.closed_bets), default=0), 2
            ),
            "worst_trade": round(
                min((b.pnl for b in self.state.closed_bets), default=0), 2
            ),
        }

    def _save_state(self):
        """Persist state to JSON file."""
        try:
            data = {
                "cash": self.state.cash,
                "total_deposited": self.state.total_deposited,
                "bets": [asdict(b) for b in self.state.bets],
                "closed_bets": [asdict(b) for b in self.state.closed_bets],
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving bankroll state: {e}")

    def _load_state(self) -> BankrollState:
        """Load state from JSON file."""
        if not os.path.exists(self.state_file):
            return BankrollState()

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            state = BankrollState(
                cash=data.get("cash", config.BANKROLL),
                total_deposited=data.get("total_deposited", config.BANKROLL),
            )
            state.bets = [Bet(**b) for b in data.get("bets", [])]
            state.closed_bets = [Bet(**b) for b in data.get("closed_bets", [])]

            logger.info(
                f"Loaded bankroll: ${state.total_value:.2f} "
                f"({len(state.open_bets)} open, {len(state.closed_bets)} closed)"
            )
            return state

        except Exception as e:
            logger.error(f"Error loading bankroll state: {e}")
            return BankrollState()
