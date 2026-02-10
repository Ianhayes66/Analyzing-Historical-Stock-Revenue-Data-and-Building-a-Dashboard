"""
Trading Bot Configuration
-------------------------
Central configuration for all bot parameters. Adjust these to tune
strategy behavior, risk limits, and operational settings.
"""

# ── Watchlist ───────────────────────────────────────────────────────────
# Tickers the bot monitors. Add/remove as needed.
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "SPY",
]

# ── Data Settings ───────────────────────────────────────────────────────
DATA_INTERVAL = "1d"          # Candle interval: 1m, 5m, 15m, 1h, 1d
DATA_PERIOD = "1y"            # Look-back period for historical data
BACKTEST_PERIOD = "2y"        # Look-back for backtesting

# ── Technical Indicator Parameters ──────────────────────────────────────
# EMA
EMA_SHORT = 9
EMA_LONG = 21
EMA_TREND = 50               # Trend-filter EMA

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# ATR (Average True Range) — used for stop-loss sizing
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0    # Stop-loss at 2x ATR from entry

# VWAP — intraday only, ignored on daily data
VWAP_ENABLED = True

# ADX (trend strength)
ADX_PERIOD = 14
ADX_THRESHOLD = 25            # Minimum ADX to confirm a trend

# Stochastic RSI
STOCH_RSI_PERIOD = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3

# ── Strategy Weights ────────────────────────────────────────────────────
# The signal aggregator scores each strategy; weights control influence.
STRATEGY_WEIGHTS = {
    "momentum":       0.30,
    "mean_reversion": 0.25,
    "breakout":       0.25,
    "trend_follow":   0.20,
}

# Minimum aggregated confidence score to trigger a trade (0-1)
MIN_SIGNAL_CONFIDENCE = 0.60

# ── Risk Management ────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.00  # Starting paper-trade capital ($)
MAX_POSITION_PCT = 0.05       # Max 5% of portfolio in any single position
MAX_PORTFOLIO_RISK = 0.02     # Max 2% portfolio risk per trade
MAX_OPEN_POSITIONS = 10       # Maximum concurrent positions
MAX_DAILY_LOSS_PCT = 0.03     # Stop trading if daily loss exceeds 3%
TRAILING_STOP_PCT = 0.05      # 5% trailing stop on open positions

# ── Execution ───────────────────────────────────────────────────────────
PAPER_TRADE = True            # True = simulation, False = live (requires broker API)
TRADE_LOG_FILE = "trades.csv"
REBALANCE_INTERVAL_MINUTES = 60  # How often to re-evaluate positions

# ── Dashboard ───────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True
