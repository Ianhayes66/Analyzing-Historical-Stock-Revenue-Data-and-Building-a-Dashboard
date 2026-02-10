# Multi-Strategy Trading Bot

A Python-based algorithmic trading bot that combines four distinct strategies through a weighted signal aggregator, enforces strict risk management, and provides a real-time performance dashboard.

> **Disclaimer**: This bot is for educational and paper-trading purposes. No trading system guarantees profits. Always paper-trade extensively before considering real capital, and never risk money you cannot afford to lose.

## Architecture

```
main.py            CLI entry point (run, backtest, scan, dashboard)
├── engine.py      Orchestrator: data → indicators → strategies → trades
├── data_fetcher.py   Market data from Yahoo Finance (free, no API key)
├── indicators.py     Technical indicator computation (20+ indicators)
├── strategies.py     Four independent strategy modules
├── risk_manager.py   Position sizing, stop-losses, portfolio limits
├── backtester.py     Walk-forward historical simulation
├── dashboard.py      Real-time Plotly Dash web dashboard
└── config.py         All tunable parameters in one place
```

## Strategies

| Strategy | Edge | Key Indicators |
|---|---|---|
| **Momentum** | Rides acceleration in price trends | EMA crossovers, MACD histogram, RSI, Rate of Change, volume confirmation |
| **Mean Reversion** | Buys oversold / sells overbought in ranging markets | Bollinger Bands %B, RSI extremes, Stochastic RSI, ADX filter |
| **Breakout** | Catches new trends as price breaks key levels | Support/Resistance, volume surge, BB width expansion, ADX confirmation |
| **Trend Following** | Stays in established trends until reversal | ADX strength, +DI/-DI crossover, EMA alignment, OBV trend |

Signals are combined via a **weighted ensemble** — each strategy votes BUY/SELL/HOLD with a confidence score (0-1), multiplied by its configured weight. A trade only executes when the aggregated score exceeds the minimum confidence threshold (default 60%).

## Risk Management

- **ATR-based position sizing**: Risk per trade = 2% of portfolio, sized by volatility
- **ATR-based stop-losses**: Hard stop at 2x ATR below entry
- **Trailing stops**: 5% trailing stop ratchets in profitable direction
- **Position cap**: Max 5% of portfolio in any single stock
- **Max positions**: 10 concurrent positions
- **Daily circuit breaker**: Stops all trading if daily loss exceeds 3%

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Scan
Scan all watchlist tickers and show current signals:
```bash
python main.py scan
```

### Backtest
Test strategies on historical data before risking capital:
```bash
# Backtest all strategies on full watchlist (2 years)
python main.py backtest

# Backtest specific tickers
python main.py backtest -t AAPL TSLA NVDA

# Backtest a single strategy
python main.py backtest -s momentum

# Custom period
python main.py backtest -p 5y
```

### Run the Bot (Paper Trading)
```bash
# Start the bot (paper-trade mode by default)
python main.py run

# Start with the web dashboard
python main.py run --with-dashboard
```

### Dashboard Only
```bash
python main.py dashboard
# Open http://localhost:8050 in your browser
```

## Configuration

All parameters are in `config.py`. Key settings to customize:

| Setting | Default | Description |
|---|---|---|
| `WATCHLIST` | 10 large-caps | Tickers to monitor |
| `INITIAL_CAPITAL` | $100,000 | Starting paper capital |
| `MIN_SIGNAL_CONFIDENCE` | 0.60 | Minimum score to trigger a trade |
| `MAX_PORTFOLIO_RISK` | 0.02 | Max 2% risk per trade |
| `MAX_POSITION_PCT` | 0.05 | Max 5% in any single position |
| `ATR_STOP_MULTIPLIER` | 2.0 | Stop-loss distance in ATR multiples |
| `STRATEGY_WEIGHTS` | See config | Relative weight of each strategy |

## Output Files

- `trades.csv` — Log of all executed trades
- `trading_bot.log` — Full application log
