"""
Trading Engine
---------------
The core orchestrator that ties everything together:
  1. Fetches market data
  2. Computes indicators
  3. Runs all strategies
  4. Aggregates signals with confidence weighting
  5. Passes decisions through risk management
  6. Executes paper trades (or live trades if configured)
  7. Logs everything

This is the main loop that runs on a schedule.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config
import data_fetcher
import indicators
from risk_manager import RiskManager, TradeRecord
from strategies import STRATEGY_REGISTRY, Signal

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that orchestrates data -> signals -> trades."""

    def __init__(self, risk_manager: Optional[RiskManager] = None):
        self.rm = risk_manager or RiskManager()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.latest_signals: Dict[str, List[Signal]] = {}
        self.iteration = 0

    def run_cycle(self) -> Dict[str, dict]:
        """
        Execute one full trading cycle:
          fetch -> indicators -> strategies -> risk check -> execute
        Returns a summary dict per ticker.
        """
        self.iteration += 1
        logger.info(f"=== Trading Cycle #{self.iteration} ===")

        # Step 1: Fetch data
        self.market_data = data_fetcher.fetch_all()
        if not self.market_data:
            logger.error("No market data available. Skipping cycle.")
            return {}

        # Step 2: Check stops on existing positions
        current_prices = self._get_current_prices()
        stopped_out = self.rm.check_stops(current_prices)
        if stopped_out:
            logger.info(f"Stopped out of: {stopped_out}")

        cycle_results = {}

        for ticker, raw_df in self.market_data.items():
            try:
                result = self._process_ticker(ticker, raw_df)
                cycle_results[ticker] = result
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)
                cycle_results[ticker] = {"ticker": ticker, "error": str(e)}

        # Log portfolio state
        summary = self.rm.get_portfolio_summary(current_prices)
        logger.info(
            f"Portfolio: ${summary['portfolio_value']:,.2f} | "
            f"Cash: ${summary['cash']:,.2f} | "
            f"Open: {summary['open_positions']} | "
            f"Return: {summary['return_pct']:.2%}"
        )

        return cycle_results

    def _process_ticker(self, ticker: str, raw_df: pd.DataFrame) -> dict:
        """Process a single ticker: indicators -> strategy signals -> trade decision."""
        # Compute indicators
        df = indicators.compute_all(raw_df)
        df = df.dropna()

        if len(df) < 60:
            return {"ticker": ticker, "action": "SKIP", "reason": "Insufficient data"}

        last_row = df.iloc[-1]
        current_price = last_row["close"]
        atr = last_row["atr"]

        # Run all strategies
        signals = []
        for name, func in STRATEGY_REGISTRY.items():
            signal = func(df)
            signals.append(signal)

        self.latest_signals[ticker] = signals

        # Aggregate signals
        direction, confidence, reasons = self._aggregate_signals(signals)

        result = {
            "ticker": ticker,
            "price": current_price,
            "direction": direction,
            "confidence": confidence,
            "reasons": reasons,
            "action": "HOLD",
        }

        # Execute if confidence meets threshold
        if direction == "BUY" and confidence >= config.MIN_SIGNAL_CONFIDENCE:
            result["action"] = self._execute_buy(ticker, current_price, atr)

        elif direction == "SELL" and confidence >= config.MIN_SIGNAL_CONFIDENCE:
            result["action"] = self._execute_sell(ticker, current_price)

        # Log signal
        self._log_signal(result)
        return result

    def _aggregate_signals(
        self, signals: List[Signal]
    ) -> Tuple[str, float, List[str]]:
        """
        Weighted aggregation of strategy signals.
        Returns (direction, confidence, list_of_reasons).
        """
        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        for signal in signals:
            weight = config.STRATEGY_WEIGHTS.get(signal.strategy, 0.25)
            if signal.direction == "BUY":
                buy_score += signal.confidence * weight
                reasons.append(f"[{signal.strategy}] BUY {signal.confidence:.0%}: {signal.reason}")
            elif signal.direction == "SELL":
                sell_score += signal.confidence * weight
                reasons.append(f"[{signal.strategy}] SELL {signal.confidence:.0%}: {signal.reason}")

        if buy_score > sell_score and buy_score > 0:
            return "BUY", buy_score, reasons
        elif sell_score > buy_score and sell_score > 0:
            return "SELL", sell_score, reasons
        return "HOLD", 0.0, reasons

    def _execute_buy(self, ticker: str, price: float, atr: float) -> str:
        """Attempt to open a long position."""
        shares = self.rm.calculate_position_size(ticker, price, atr)
        if shares <= 0:
            return "SKIP (position too small)"

        pos = self.rm.open_position(
            ticker, "LONG", price, shares, atr, strategy="ensemble"
        )
        if pos:
            self._write_trade_log(ticker, "BUY", price, shares)
            return f"BUY {shares} shares @ ${price:.2f}"
        return "BLOCKED by risk manager"

    def _execute_sell(self, ticker: str, price: float) -> str:
        """Attempt to close an existing long position."""
        if ticker not in self.rm.positions:
            return "HOLD (no position to sell)"

        record = self.rm.close_position(ticker, price, reason="Sell signal")
        if record:
            self._write_trade_log(
                ticker, "SELL", price, record.shares, pnl=record.pnl
            )
            return f"SELL @ ${price:.2f} | PnL: ${record.pnl:,.2f}"
        return "CLOSE FAILED"

    def _get_current_prices(self) -> Dict[str, float]:
        """Get latest close price for all held tickers."""
        prices = {}
        for ticker, df in self.market_data.items():
            if not df.empty:
                prices[ticker] = df.iloc[-1]["close"]
        return prices

    def _log_signal(self, result: dict):
        """Log a signal/action to the console."""
        ticker = result["ticker"]
        action = result.get("action", "HOLD")
        confidence = result.get("confidence", 0)
        price = result.get("price", 0)
        logger.info(
            f"  {ticker:<6} | ${price:>10.2f} | {action:<30} | conf={confidence:.0%}"
        )

    def _write_trade_log(
        self,
        ticker: str,
        side: str,
        price: float,
        shares: int,
        pnl: float = 0.0,
    ):
        """Append trade to CSV log."""
        file_exists = os.path.exists(config.TRADE_LOG_FILE)
        with open(config.TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["timestamp", "ticker", "side", "price", "shares", "pnl"]
                )
            writer.writerow([
                datetime.now().isoformat(),
                ticker,
                side,
                f"{price:.2f}",
                shares,
                f"{pnl:.2f}",
            ])

    def get_status(self) -> dict:
        """Return complete engine status for the dashboard."""
        current_prices = self._get_current_prices()
        portfolio = self.rm.get_portfolio_summary(current_prices)
        positions = {}
        for ticker, pos in self.rm.positions.items():
            price = current_prices.get(ticker, pos.entry_price)
            positions[ticker] = {
                "direction": pos.direction,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "current_price": price,
                "pnl": pos.current_pnl(price),
                "pnl_pct": pos.current_pnl_pct(price),
                "stop_loss": pos.stop_loss,
                "trailing_stop": pos.trailing_stop,
            }

        return {
            "iteration": self.iteration,
            "portfolio": portfolio,
            "positions": positions,
            "trade_history": [
                {
                    "ticker": t.ticker,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "reason": t.reason,
                }
                for t in self.rm.trade_history[-50:]  # Last 50 trades
            ],
            "latest_signals": {
                ticker: [
                    {"strategy": s.strategy, "direction": s.direction,
                     "confidence": s.confidence, "reason": s.reason}
                    for s in signals
                ]
                for ticker, signals in self.latest_signals.items()
            },
        }
