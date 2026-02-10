#!/usr/bin/env python3
"""
Polymarket Sentiment Predictor
================================
A prediction market bot that finds edges by analyzing public sentiment
from Reddit and Twitter, then comparing it to Polymarket prices.

Commands:
    python main.py scan           Scan markets for opportunities
    python main.py monitor        Continuous monitoring mode
    python main.py dashboard      Launch web dashboard
    python main.py bet            Scan + auto-place best bets
    python main.py status         Show bankroll status
    python main.py history        Show trade history

The alpha: when public sentiment diverges from market prices, we
exploit the gap using Bayesian probability updating and Kelly
Criterion bet sizing.
"""

import argparse
import logging
import signal
import sys
import threading
import time

import schedule

import config
from polymarket_predictor.bankroll_manager import BankrollManager
from polymarket_predictor.signal_generator import SignalGenerator

# ── Logging ─────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("predictor.log", mode="a"),
        ],
    )


logger = logging.getLogger(__name__)


# ── Commands ────────────────────────────────────────────────────────────

def cmd_scan(args):
    """One-time scan: find opportunities and display them."""
    setup_logging(args.verbose)
    logger.info("Starting market scan...")

    gen = SignalGenerator()
    opportunities = gen.scan()
    gen.print_opportunities(opportunities)

    if opportunities:
        print("\nTop opportunity details:")
        best = opportunities[0]
        print(f"  Market:     {best.market['question']}")
        print(f"  Direction:  {best.signal.direction}")
        print(f"  Edge:       {best.signal.edge_estimate:+.1%}")
        print(f"  Confidence: {best.signal.confidence:.0%}")
        print(f"  Signal:     {best.signal.signal_type}")
        print(f"  Bet Size:   ${best.recommended_bet:.2f}")
        print(f"\n  Sentiment:  {best.sentiment.yes_probability:.1%} YES "
              f"(n={best.sentiment.sample_size})")
        print(f"  Market:     {best.signal.market_price:.1%} YES")
        print(f"  Bayesian:   {best.signal.bayesian_posterior:.1%} YES")
        print(f"\n  Echo Chamber: {'YES' if best.echo.is_echo_chamber else 'No'} "
              f"(uniformity={best.echo.uniformity_score:.2f})")
        print(f"  Narrative:  {best.narrative.phase} "
              f"(velocity={best.narrative.velocity:+.3f})")
        print(f"\n  Explanation: {best.signal.explanation}")


def cmd_monitor(args):
    """Continuous monitoring: scan every N minutes and alert on opportunities."""
    setup_logging(args.verbose)
    logger.info(f"Starting continuous monitor (every {config.SCAN_INTERVAL_MINUTES} min)")

    gen = SignalGenerator()

    # Optionally launch dashboard
    if args.with_dashboard:
        from dashboard import run_dashboard
        dash_thread = threading.Thread(
            target=run_dashboard, args=(gen,), daemon=True
        )
        dash_thread.start()
        logger.info(f"Dashboard at http://localhost:{config.DASHBOARD_PORT}")

    def _scan_cycle():
        opportunities = gen.scan()
        gen.print_opportunities(opportunities)
        # Alert on strong signals
        for opp in opportunities[:3]:
            if opp.composite_score > 0.7:
                print(f"\n{'!'*60}")
                print(f"STRONG SIGNAL: {opp.signal.direction} on "
                      f"'{opp.market['question'][:60]}'")
                print(f"Edge: {opp.signal.edge_estimate:+.1%} | "
                      f"Recommended bet: ${opp.recommended_bet:.2f}")
                print(f"{'!'*60}\n")

    # First scan immediately
    _scan_cycle()

    # Schedule recurring scans
    schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(_scan_cycle)

    running = True
    def _stop(signum, frame):
        nonlocal running
        logger.info("Stopping monitor...")
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        schedule.run_pending()
        time.sleep(1)


def cmd_bet(args):
    """Scan and auto-place bets on the best opportunities."""
    setup_logging(args.verbose)

    gen = SignalGenerator()
    opportunities = gen.scan()
    gen.print_opportunities(opportunities)

    if not opportunities:
        print("\nNo opportunities found. Nothing to bet on.")
        return

    # Filter to only bettable opportunities
    bettable = [o for o in opportunities if o.recommended_bet >= config.MIN_BET_SIZE]

    if not bettable:
        print("\nOpportunities found but none meet betting criteria.")
        return

    # Show what we'd bet on
    print(f"\n{'='*60}")
    print(f"PROPOSED BETS ({len(bettable[:config.MAX_CONCURRENT_BETS])} bets)")
    print(f"{'='*60}")

    total_bet = 0
    for i, opp in enumerate(bettable[:config.MAX_CONCURRENT_BETS], 1):
        direction = "YES" if opp.signal.direction == "BUY_YES" else "NO"
        price = opp.signal.market_price if direction == "YES" else (1 - opp.signal.market_price)

        print(f"\n  Bet #{i}:")
        print(f"    Market:    {opp.market['question'][:70]}")
        print(f"    Direction: BUY {direction}")
        print(f"    Amount:    ${opp.recommended_bet:.2f}")
        print(f"    Price:     {price:.2f} per share")
        print(f"    Edge:      {opp.signal.edge_estimate:+.1%}")
        print(f"    Signal:    {opp.signal.signal_type} "
              f"(confidence={opp.signal.confidence:.0%})")
        total_bet += opp.recommended_bet

    bankroll = gen.bankroll.get_summary()
    print(f"\n  Total bet: ${total_bet:.2f} of ${bankroll['cash']:.2f} cash")

    if args.auto:
        # Auto-place without confirmation
        _execute_bets(gen, bettable[:config.MAX_CONCURRENT_BETS])
    else:
        print(f"\n  Run with --auto to place these bets automatically.")
        print(f"  Or manually place on Polymarket using the signals above.")


def _execute_bets(gen: SignalGenerator, opportunities: list):
    """Place bets via the bankroll manager."""
    for opp in opportunities:
        direction = "YES" if opp.signal.direction == "BUY_YES" else "NO"
        price = opp.signal.market_price if direction == "YES" else (1 - opp.signal.market_price)

        bet = gen.bankroll.place_bet(
            market_id=opp.market["id"],
            question=opp.market["question"],
            direction=direction,
            amount=opp.recommended_bet,
            price=price,
            edge=opp.signal.edge_estimate,
            confidence=opp.signal.confidence,
            signal_type=opp.signal.signal_type,
        )
        if bet:
            print(f"  PLACED: ${bet.amount:.2f} on {direction} "
                  f"'{opp.market['question'][:50]}'")
        else:
            print(f"  BLOCKED: Could not place bet on "
                  f"'{opp.market['question'][:50]}'")


def cmd_dashboard(args):
    """Launch the web dashboard."""
    setup_logging(args.verbose)
    from dashboard import run_dashboard

    gen = SignalGenerator()
    # Do initial scan
    gen.scan()
    run_dashboard(gen, debug=True)


def cmd_status(args):
    """Show bankroll status."""
    setup_logging(False)
    mgr = BankrollManager()
    summary = mgr.get_summary()

    print(f"\n{'='*50}")
    print(f"{'BANKROLL STATUS':^50}")
    print(f"{'='*50}")
    print(f"  Total Value:    ${summary['bankroll']:>10.2f}")
    print(f"  Cash:           ${summary['cash']:>10.2f}")
    print(f"  Invested:       ${summary['invested']:>10.2f}")
    print(f"  Total P&L:      ${summary['total_pnl']:>+10.2f}")
    print(f"  ROI:            {summary['roi_pct']:>+9.1f}%")
    print(f"  Win Rate:       {summary['win_rate']:>9.0f}%")
    print(f"  Open Bets:      {summary['open_bets']:>10}")
    print(f"  Closed Bets:    {summary['closed_bets']:>10}")
    print(f"  Best Trade:     ${summary['best_trade']:>+10.2f}")
    print(f"  Worst Trade:    ${summary['worst_trade']:>+10.2f}")
    print(f"{'='*50}")

    # Show open bets
    if mgr.state.open_bets:
        print(f"\nOpen Bets:")
        for b in mgr.state.open_bets:
            print(f"  {b.direction:>4} ${b.amount:.2f} @ {b.entry_price:.2f} "
                  f"| Edge: {b.estimated_edge:+.1%} "
                  f"| {b.question[:50]}")


def cmd_history(args):
    """Show trade history."""
    setup_logging(False)
    mgr = BankrollManager()

    if not mgr.state.closed_bets:
        print("\nNo trade history yet.")
        return

    print(f"\n{'='*90}")
    print(f"{'TRADE HISTORY':^90}")
    print(f"{'='*90}")
    print(f"{'Dir':>5} {'Amount':>8} {'Entry':>7} {'Exit':>7} "
          f"{'P&L':>9} {'Status':>8} {'Type':<14} {'Question':<30}")
    print(f"{'-'*90}")

    for b in reversed(mgr.state.closed_bets):
        pnl_str = f"${b.pnl:+.2f}"
        print(f"{b.direction:>5} ${b.amount:>7.2f} {b.entry_price:>7.2f} "
              f"{b.exit_price:>7.2f} {pnl_str:>9} {b.status:>8} "
              f"{b.signal_type:<14} {b.question[:28]}")

    summary = mgr.get_summary()
    print(f"{'-'*90}")
    print(f"Total: {len(mgr.state.closed_bets)} trades | "
          f"P&L: ${summary['total_pnl']:+.2f} | "
          f"Win Rate: {summary['win_rate']:.0f}% | "
          f"ROI: {summary['roi_pct']:+.1f}%")
    print(f"{'='*90}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Sentiment Predictor — $50 Empire Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                    # One-time market scan
  python main.py scan -v                 # Verbose scan with debug output
  python main.py monitor                 # Continuous monitoring
  python main.py monitor --with-dashboard  # Monitor + web dashboard
  python main.py bet                     # Scan + show proposed bets
  python main.py bet --auto              # Scan + auto-place bets
  python main.py dashboard              # Web dashboard only
  python main.py status                 # Show bankroll
  python main.py history                # Show trade history

Setup:
  1. Copy .env.example to .env
  2. Add Reddit API credentials (free at reddit.com/prefs/apps)
  3. Optionally add Twitter Bearer Token
  4. pip install -r requirements.txt
  5. python main.py scan
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # scan
    subparsers.add_parser("scan", help="One-time market scan")

    # monitor
    mon = subparsers.add_parser("monitor", help="Continuous monitoring")
    mon.add_argument("--with-dashboard", action="store_true",
                     help="Also launch web dashboard")

    # bet
    bet = subparsers.add_parser("bet", help="Scan + propose/place bets")
    bet.add_argument("--auto", action="store_true",
                     help="Auto-place bets without confirmation")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch web dashboard")

    # status
    subparsers.add_parser("status", help="Show bankroll status")

    # history
    subparsers.add_parser("history", help="Show trade history")

    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "monitor": cmd_monitor,
        "bet": cmd_bet,
        "dashboard": cmd_dashboard,
        "status": cmd_status,
        "history": cmd_history,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
