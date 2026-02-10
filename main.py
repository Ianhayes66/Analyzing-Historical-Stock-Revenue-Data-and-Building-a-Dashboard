#!/usr/bin/env python3
"""
Trading Bot — Main Entry Point
================================
Usage:
    python main.py run          Run the live paper-trading bot
    python main.py backtest     Backtest strategies on historical data
    python main.py dashboard    Launch the performance dashboard only
    python main.py scan         One-time scan of all watchlist tickers
    python main.py status       Show current portfolio status

The bot uses a multi-strategy ensemble approach combining:
  - Momentum (EMA crossovers, MACD, RSI, ROC)
  - Mean Reversion (Bollinger Bands, RSI extremes, StochRSI)
  - Breakout (Support/Resistance, volume surges)
  - Trend Following (ADX, directional index, EMA alignment)

All strategies feed into a weighted signal aggregator, and trades are
filtered through a strict risk management layer with ATR-based stops,
position-size limits, and daily drawdown circuit breakers.
"""

import argparse
import logging
import signal
import sys
import threading
import time

import schedule

import config
import data_fetcher
from backtester import print_backtest_summary, run_backtest, run_multi_ticker_backtest
from dashboard import run_dashboard
from engine import TradingEngine
from risk_manager import RiskManager

# ── Logging Setup ───────────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("trading_bot.log", mode="a"),
        ],
    )

logger = logging.getLogger(__name__)


# ── Commands ────────────────────────────────────────────────────────────

def cmd_run(args):
    """Run the live (paper) trading bot on a schedule."""
    setup_logging()
    logger.info("Starting Trading Bot in PAPER TRADE mode")
    logger.info(f"Watchlist: {config.WATCHLIST}")
    logger.info(f"Interval: {config.DATA_INTERVAL} | Rebalance every "
                f"{config.REBALANCE_INTERVAL_MINUTES} min")

    engine = TradingEngine()

    # Run first cycle immediately
    engine.run_cycle()

    # Schedule subsequent cycles
    schedule.every(config.REBALANCE_INTERVAL_MINUTES).minutes.do(engine.run_cycle)

    # Optionally launch dashboard in background
    if args.with_dashboard:
        dash_thread = threading.Thread(
            target=run_dashboard, args=(engine,), daemon=True
        )
        dash_thread.start()
        logger.info("Dashboard thread started")

    # Graceful shutdown
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received. Stopping...")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while running:
        schedule.run_pending()
        time.sleep(1)

    logger.info("Bot stopped. Final portfolio summary:")
    summary = engine.rm.get_portfolio_summary()
    _print_summary(summary)


def cmd_backtest(args):
    """Run backtests on historical data."""
    setup_logging()
    logger.info("Running Backtest")
    logger.info(f"Tickers: {args.tickers or config.WATCHLIST}")
    logger.info(f"Period: {args.period}")

    tickers = args.tickers if args.tickers else config.WATCHLIST
    data = data_fetcher.fetch_all(
        tickers=tickers, period=args.period, interval=config.DATA_INTERVAL
    )

    if not data:
        logger.error("No data fetched. Exiting.")
        return

    strategy = args.strategy if args.strategy != "all" else None
    results = run_multi_ticker_backtest(data, strategy_name=strategy)
    print_backtest_summary(results)


def cmd_dashboard(args):
    """Launch the dashboard standalone."""
    setup_logging()
    logger.info("Launching Dashboard (standalone mode)")
    engine = TradingEngine()
    # Do one initial scan so the dashboard has data
    engine.run_cycle()
    run_dashboard(engine, debug=True)


def cmd_scan(args):
    """One-time scan: fetch data, compute signals, print results."""
    setup_logging()
    logger.info("Running one-time market scan")

    engine = TradingEngine()
    results = engine.run_cycle()

    print("\n" + "=" * 80)
    print(f"{'MARKET SCAN RESULTS':^80}")
    print("=" * 80)
    print(f"{'Ticker':<8} {'Price':>10} {'Signal':>8} {'Conf':>7} {'Action':<30}")
    print("-" * 80)

    for ticker, r in sorted(results.items()):
        if "error" in r:
            print(f"{ticker:<8} {'ERROR':>10}")
            continue
        print(
            f"{r['ticker']:<8} ${r.get('price', 0):>9.2f} "
            f"{r.get('direction', 'HOLD'):>8} {r.get('confidence', 0):>6.0%} "
            f"{r.get('action', 'HOLD'):<30}"
        )

    print("=" * 80)

    # Print reasons for top signals
    print("\nSignal Details:")
    for ticker, r in sorted(results.items()):
        reasons = r.get("reasons", [])
        if reasons and r.get("direction") != "HOLD":
            print(f"\n  {ticker} ({r.get('direction')}):")
            for reason in reasons:
                print(f"    {reason}")


def cmd_status(args):
    """Show current portfolio status."""
    setup_logging()
    engine = TradingEngine()
    summary = engine.rm.get_portfolio_summary()
    _print_summary(summary)


def _print_summary(summary: dict):
    print("\n" + "=" * 60)
    print(f"{'PORTFOLIO SUMMARY':^60}")
    print("=" * 60)
    print(f"  Portfolio Value:  ${summary['portfolio_value']:>12,.2f}")
    print(f"  Cash:             ${summary['cash']:>12,.2f}")
    print(f"  Open Positions:   {summary['open_positions']:>12}")
    print(f"  Total Trades:     {summary['total_trades']:>12}")
    print(f"  Win Rate:         {summary['win_rate']:>11.0%}")
    print(f"  Realized P&L:     ${summary['total_realized_pnl']:>12,.2f}")
    print(f"  Unrealized P&L:   ${summary['unrealized_pnl']:>12,.2f}")
    print(f"  Profit Factor:    {summary['profit_factor']:>12.2f}")
    print(f"  Total Return:     {summary['return_pct']:>11.2%}")
    print("=" * 60)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Strategy Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                      # Quick scan of all watchlist tickers
  python main.py backtest                  # Backtest all strategies on watchlist
  python main.py backtest -t AAPL TSLA     # Backtest specific tickers
  python main.py backtest -s momentum      # Backtest only momentum strategy
  python main.py run                       # Start paper-trading bot
  python main.py run --with-dashboard      # Start bot + web dashboard
  python main.py dashboard                 # Launch dashboard only
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = subparsers.add_parser("run", help="Start the trading bot")
    run_parser.add_argument(
        "--with-dashboard", action="store_true",
        help="Also launch the web dashboard",
    )

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtests")
    bt_parser.add_argument(
        "-t", "--tickers", nargs="+", default=None,
        help="Tickers to backtest (default: watchlist)",
    )
    bt_parser.add_argument(
        "-p", "--period", default=config.BACKTEST_PERIOD,
        help=f"Historical period (default: {config.BACKTEST_PERIOD})",
    )
    bt_parser.add_argument(
        "-s", "--strategy", default="all",
        choices=["all", "momentum", "mean_reversion", "breakout", "trend_follow"],
        help="Strategy to backtest (default: all ensemble)",
    )

    # dashboard
    subparsers.add_parser("dashboard", help="Launch dashboard")

    # scan
    subparsers.add_parser("scan", help="One-time market scan")

    # status
    subparsers.add_parser("status", help="Show portfolio status")

    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "backtest": cmd_backtest,
        "dashboard": cmd_dashboard,
        "scan": cmd_scan,
        "status": cmd_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
