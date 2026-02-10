"""
Risk Management Module
-----------------------
Enforces position sizing, stop-losses, portfolio-level risk limits,
and trailing stops. This is the safety layer that prevents catastrophic
losses regardless of what signals the strategies generate.

Key principles:
  - Never risk more than MAX_PORTFOLIO_RISK (2%) per trade
  - ATR-based stop-losses adapt to each asset's volatility
  - Daily drawdown circuit breaker stops all trading
  - Position sizes are Kelly-criterion-inspired but capped conservatively
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    direction: str           # "LONG" or "SHORT"
    entry_price: float
    shares: int
    entry_date: datetime
    stop_loss: float
    trailing_stop: float
    highest_price: float     # Track for trailing stop
    lowest_price: float      # Track for short trailing stop

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def current_pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - current_price) * self.shares

    def current_pnl_pct(self, current_price: float) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.current_pnl(current_price) / self.cost_basis

    def update_trailing_stop(self, current_price: float):
        """Ratchet the trailing stop in the profitable direction."""
        if self.direction == "LONG":
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.trailing_stop = current_price * (1 - config.TRAILING_STOP_PCT)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                self.trailing_stop = current_price * (1 + config.TRAILING_STOP_PCT)

    def should_stop_out(self, current_price: float) -> bool:
        """Check if either hard stop or trailing stop has been hit."""
        if self.direction == "LONG":
            return current_price <= self.stop_loss or current_price <= self.trailing_stop
        else:
            return current_price >= self.stop_loss or current_price >= self.trailing_stop


@dataclass
class TradeRecord:
    ticker: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    entry_date: datetime
    exit_date: datetime
    pnl: float
    pnl_pct: float
    strategy: str
    reason: str


class RiskManager:
    """Portfolio-level risk management and position sizing."""

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.daily_pnl_date: Optional[date] = None

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def portfolio_value(self) -> float:
        """Cash + estimated value of all open positions (at entry price)."""
        pos_value = sum(p.cost_basis for p in self.positions.values())
        return self.cash + pos_value

    def calculate_position_size(
        self, ticker: str, entry_price: float, atr: float
    ) -> int:
        """
        Calculate the number of shares using ATR-based risk sizing.

        Logic:
          risk_per_share = ATR * ATR_STOP_MULTIPLIER
          max_risk_amount = portfolio_value * MAX_PORTFOLIO_RISK
          shares_by_risk  = max_risk_amount / risk_per_share
          shares_by_size  = (portfolio_value * MAX_POSITION_PCT) / entry_price
          final_shares    = min(shares_by_risk, shares_by_size)
        """
        if entry_price <= 0 or atr <= 0:
            return 0

        risk_per_share = atr * config.ATR_STOP_MULTIPLIER
        if risk_per_share <= 0:
            return 0

        pv = self.portfolio_value

        # Risk-based sizing
        max_risk = pv * config.MAX_PORTFOLIO_RISK
        shares_by_risk = int(max_risk / risk_per_share)

        # Position-cap sizing
        max_position_value = pv * config.MAX_POSITION_PCT
        shares_by_size = int(max_position_value / entry_price)

        # Cash constraint
        shares_by_cash = int(self.cash / entry_price)

        shares = max(min(shares_by_risk, shares_by_size, shares_by_cash), 0)
        logger.debug(
            f"Position sizing for {ticker}: risk={shares_by_risk}, "
            f"cap={shares_by_size}, cash={shares_by_cash} -> {shares} shares"
        )
        return shares

    def can_open_position(self, ticker: str) -> tuple:
        """Check if a new position is allowed. Returns (bool, reason)."""
        if ticker in self.positions:
            return False, f"Already holding {ticker}"

        if self.open_position_count >= config.MAX_OPEN_POSITIONS:
            return False, f"Max positions reached ({config.MAX_OPEN_POSITIONS})"

        # Daily loss circuit breaker
        self._update_daily_pnl_date()
        if self.daily_pnl < -(self.portfolio_value * config.MAX_DAILY_LOSS_PCT):
            return False, f"Daily loss limit hit ({self.daily_pnl:.2f})"

        if self.cash < 100:
            return False, "Insufficient cash"

        return True, "OK"

    def open_position(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        shares: int,
        atr: float,
        strategy: str,
    ) -> Optional[Position]:
        """Open a new position with automatic stop-loss placement."""
        allowed, reason = self.can_open_position(ticker)
        if not allowed:
            logger.warning(f"Cannot open {ticker}: {reason}")
            return None

        if shares <= 0:
            return None

        # ATR-based hard stop
        stop_distance = atr * config.ATR_STOP_MULTIPLIER
        if direction == "LONG":
            stop_loss = entry_price - stop_distance
            trailing_stop = entry_price * (1 - config.TRAILING_STOP_PCT)
        else:
            stop_loss = entry_price + stop_distance
            trailing_stop = entry_price * (1 + config.TRAILING_STOP_PCT)

        cost = entry_price * shares
        self.cash -= cost

        pos = Position(
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            shares=shares,
            entry_date=datetime.now(),
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        self.positions[ticker] = pos

        logger.info(
            f"OPENED {direction} {shares} x {ticker} @ {entry_price:.2f} "
            f"| Stop: {stop_loss:.2f} | Cost: ${cost:,.2f}"
        )
        return pos

    def close_position(
        self, ticker: str, exit_price: float, reason: str = "", strategy: str = ""
    ) -> Optional[TradeRecord]:
        """Close an existing position and record the trade."""
        if ticker not in self.positions:
            logger.warning(f"No position to close for {ticker}")
            return None

        pos = self.positions.pop(ticker)
        pnl = pos.current_pnl(exit_price)
        pnl_pct = pos.current_pnl_pct(exit_price)
        proceeds = exit_price * pos.shares
        self.cash += proceeds

        self._update_daily_pnl_date()
        self.daily_pnl += pnl

        record = TradeRecord(
            ticker=ticker,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            entry_date=pos.entry_date,
            exit_date=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=strategy,
            reason=reason,
        )
        self.trade_history.append(record)

        symbol = "+" if pnl >= 0 else ""
        logger.info(
            f"CLOSED {pos.direction} {pos.shares} x {ticker} @ {exit_price:.2f} "
            f"| PnL: {symbol}${pnl:,.2f} ({symbol}{pnl_pct:.1%}) | {reason}"
        )
        return record

    def check_stops(self, current_prices: Dict[str, float]) -> List[str]:
        """Check all open positions for stop-loss triggers."""
        tickers_to_close = []
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker)
            if price is None:
                continue
            pos.update_trailing_stop(price)
            if pos.should_stop_out(price):
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            price = current_prices[ticker]
            self.close_position(ticker, price, reason="Stop-loss triggered")

        return tickers_to_close

    def get_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None) -> dict:
        """Generate a portfolio status report."""
        total_pnl = sum(t.pnl for t in self.trade_history)
        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]
        win_rate = len(wins) / max(len(self.trade_history), 1)

        unrealized_pnl = 0.0
        if current_prices:
            for ticker, pos in self.positions.items():
                price = current_prices.get(ticker, pos.entry_price)
                unrealized_pnl += pos.current_pnl(price)

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        profit_factor = (sum(t.pnl for t in wins) / max(sum(abs(t.pnl) for t in losses), 1)) if losses else 0

        return {
            "portfolio_value": self.portfolio_value + unrealized_pnl,
            "cash": self.cash,
            "open_positions": self.open_position_count,
            "total_trades": len(self.trade_history),
            "win_rate": win_rate,
            "total_realized_pnl": total_pnl,
            "unrealized_pnl": unrealized_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "daily_pnl": self.daily_pnl,
            "return_pct": (self.portfolio_value + unrealized_pnl - self.initial_capital)
                          / self.initial_capital,
        }

    def _update_daily_pnl_date(self):
        today = date.today()
        if self.daily_pnl_date != today:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today
