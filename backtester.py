"""
Backtesting Engine
-------------------
Simulates trading strategies on historical data to measure performance
before risking real capital.

Features:
  - Walk-forward simulation (no lookahead bias)
  - Realistic transaction cost modeling
  - Drawdown analysis
  - Per-strategy and aggregate statistics
  - Equity curve generation for charting
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
import indicators
from risk_manager import RiskManager
from strategies import STRATEGY_REGISTRY, Signal

logger = logging.getLogger(__name__)

TRANSACTION_COST_PCT = 0.001  # 0.1% round-trip slippage + commission estimate


@dataclass
class BacktestResult:
    ticker: str
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_holding_days: float
    equity_curve: pd.Series
    trades: List[dict]


def run_backtest(
    ticker: str,
    df: pd.DataFrame,
    strategy_name: Optional[str] = None,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> BacktestResult:
    """
    Run a walk-forward backtest on a single ticker.

    If strategy_name is given, only that strategy is used.
    Otherwise the weighted ensemble approach is used.
    """
    # Compute indicators
    df = indicators.compute_all(df)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 60:
        logger.warning(f"Not enough data for {ticker} backtest ({len(df)} rows)")
        return _empty_result(ticker)

    rm = RiskManager(initial_capital=initial_capital)
    equity_curve = []
    trade_log = []

    strategies_to_run = (
        {strategy_name: STRATEGY_REGISTRY[strategy_name]}
        if strategy_name and strategy_name in STRATEGY_REGISTRY
        else STRATEGY_REGISTRY
    )
    weights = config.STRATEGY_WEIGHTS

    for i in range(60, len(df)):
        window = df.iloc[: i + 1]
        current_price = window.iloc[-1]["close"]
        atr = window.iloc[-1]["atr"]

        # ── Check stops on open positions ───────────────────────────────
        if ticker in rm.positions:
            pos = rm.positions[ticker]
            pos.update_trailing_stop(current_price)
            if pos.should_stop_out(current_price):
                exit_price = current_price * (1 - TRANSACTION_COST_PCT)
                record = rm.close_position(ticker, exit_price, reason="Stop-loss")
                if record:
                    trade_log.append({
                        "entry_date": record.entry_date,
                        "exit_date": record.exit_date,
                        "direction": record.direction,
                        "entry_price": record.entry_price,
                        "exit_price": record.exit_price,
                        "pnl": record.pnl,
                        "pnl_pct": record.pnl_pct,
                        "reason": record.reason,
                    })

        # ── Run strategies and aggregate signals ────────────────────────
        buy_score = 0.0
        sell_score = 0.0

        for name, func in strategies_to_run.items():
            signal = func(window)
            w = weights.get(name, 0.25) if not strategy_name else 1.0

            if signal.direction == "BUY":
                buy_score += signal.confidence * w
            elif signal.direction == "SELL":
                sell_score += signal.confidence * w

        # ── Execute based on aggregated score ───────────────────────────
        if buy_score >= config.MIN_SIGNAL_CONFIDENCE and ticker not in rm.positions:
            entry_price = current_price * (1 + TRANSACTION_COST_PCT)
            shares = rm.calculate_position_size(ticker, entry_price, atr)
            if shares > 0:
                rm.open_position(
                    ticker, "LONG", entry_price, shares, atr, strategy="ensemble"
                )

        elif sell_score >= config.MIN_SIGNAL_CONFIDENCE and ticker in rm.positions:
            exit_price = current_price * (1 - TRANSACTION_COST_PCT)
            record = rm.close_position(ticker, exit_price, reason="Sell signal")
            if record:
                trade_log.append({
                    "entry_date": record.entry_date,
                    "exit_date": record.exit_date,
                    "direction": record.direction,
                    "entry_price": record.entry_price,
                    "exit_price": record.exit_price,
                    "pnl": record.pnl,
                    "pnl_pct": record.pnl_pct,
                    "reason": record.reason,
                })

        # ── Record equity ───────────────────────────────────────────────
        unrealized = 0.0
        if ticker in rm.positions:
            unrealized = rm.positions[ticker].current_pnl(current_price)
        equity_curve.append(rm.cash + unrealized + sum(
            p.cost_basis for p in rm.positions.values()
        ))

    # ── Close any remaining position ────────────────────────────────────
    if ticker in rm.positions:
        final_price = df.iloc[-1]["close"]
        record = rm.close_position(ticker, final_price, reason="Backtest end")
        if record:
            trade_log.append({
                "entry_date": record.entry_date,
                "exit_date": record.exit_date,
                "direction": record.direction,
                "entry_price": record.entry_price,
                "exit_price": record.exit_price,
                "pnl": record.pnl,
                "pnl_pct": record.pnl_pct,
                "reason": record.reason,
            })

    # ── Compute performance metrics ─────────────────────────────────────
    eq = pd.Series(equity_curve)
    return _compute_metrics(ticker, eq, rm, trade_log, initial_capital)


def run_multi_ticker_backtest(
    data: Dict[str, pd.DataFrame],
    strategy_name: Optional[str] = None,
) -> Dict[str, BacktestResult]:
    """Run backtests for multiple tickers."""
    results = {}
    for ticker, df in data.items():
        logger.info(f"Backtesting {ticker}...")
        results[ticker] = run_backtest(ticker, df, strategy_name)
    return results


def print_backtest_summary(results: Dict[str, BacktestResult]):
    """Print a formatted summary table of backtest results."""
    print("\n" + "=" * 90)
    print(f"{'BACKTEST RESULTS':^90}")
    print("=" * 90)
    print(
        f"{'Ticker':<8} {'Return%':>9} {'Annual%':>9} {'Sharpe':>8} "
        f"{'MaxDD%':>8} {'WinRate':>8} {'Trades':>7} {'PF':>7}"
    )
    print("-" * 90)

    for ticker, r in sorted(results.items()):
        print(
            f"{ticker:<8} {r.total_return_pct:>8.1f}% {r.annual_return_pct:>8.1f}% "
            f"{r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>7.1f}% "
            f"{r.win_rate:>7.0%} {r.total_trades:>7} {r.profit_factor:>7.2f}"
        )

    # Aggregate stats
    avg_return = np.mean([r.total_return_pct for r in results.values()])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])
    avg_wr = np.mean([r.win_rate for r in results.values()])
    total_trades = sum(r.total_trades for r in results.values())
    print("-" * 90)
    print(
        f"{'AVG':<8} {avg_return:>8.1f}%           "
        f"{avg_sharpe:>8.2f}           "
        f"{avg_wr:>7.0%} {total_trades:>7}"
    )
    print("=" * 90)


# ── Helpers ─────────────────────────────────────────────────────────────

def _compute_metrics(
    ticker: str,
    equity: pd.Series,
    rm: RiskManager,
    trade_log: list,
    initial_capital: float,
) -> BacktestResult:
    if equity.empty:
        return _empty_result(ticker)

    total_return = (equity.iloc[-1] / initial_capital) - 1
    n_days = len(equity)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # Sharpe ratio (daily returns)
    daily_returns = equity.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    # Maximum drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100

    # Win rate and profit factor
    summary = rm.get_portfolio_summary()

    # Average holding period
    holding_days = []
    for t in trade_log:
        if t["entry_date"] and t["exit_date"]:
            delta = (t["exit_date"] - t["entry_date"]).total_seconds() / 86400
            holding_days.append(delta)
    avg_hold = np.mean(holding_days) if holding_days else 0

    return BacktestResult(
        ticker=ticker,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return * 100,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=summary["win_rate"],
        total_trades=summary["total_trades"],
        profit_factor=summary["profit_factor"],
        avg_holding_days=avg_hold,
        equity_curve=equity,
        trades=trade_log,
    )


def _empty_result(ticker: str) -> BacktestResult:
    return BacktestResult(
        ticker=ticker,
        total_return_pct=0,
        annual_return_pct=0,
        sharpe_ratio=0,
        max_drawdown_pct=0,
        win_rate=0,
        total_trades=0,
        profit_factor=0,
        avg_holding_days=0,
        equity_curve=pd.Series(dtype=float),
        trades=[],
    )
