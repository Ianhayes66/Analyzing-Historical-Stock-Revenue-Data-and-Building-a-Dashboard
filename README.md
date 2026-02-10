# Polymarket Sentiment Predictor

A prediction market intelligence engine that finds edges on Polymarket by analyzing public sentiment from Reddit and Twitter/X, detecting divergences between crowd opinion and market prices, and sizing bets with the Kelly Criterion.

> **Disclaimer**: Prediction markets involve real financial risk. This tool provides analysis, not guarantees. Past performance does not predict future results. Never bet more than you can afford to lose.

## How It Works — The Edge

```
                    ┌─────────────┐
                    │  Polymarket │
                    │  Live Prices│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌───────────┐    ┌──────────────┐    ┌───────────┐
  │  Reddit   │    │   Bayesian   │    │ Twitter/X │
  │ Sentiment │───▶│   Updater    │◀───│ Sentiment │
  └───────────┘    └──────┬───────┘    └───────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌─────────┐ ┌──────────┐
        │Divergence│ │  Echo   │ │Narrative │
        │ Engine   │ │Chamber  │ │Momentum  │
        └─────┬────┘ │Detector │ │ Tracker  │
              │      └────┬────┘ └────┬─────┘
              └───────────┼───────────┘
                          ▼
                   ┌─────────────┐
                   │   Kelly     │
                   │  Criterion  │
                   │ Bet Sizing  │
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │  $50 → ???  │
                   └─────────────┘
```

**The core insight**: When public sentiment on Reddit/Twitter strongly diverges from what the Polymarket price implies, one of them is wrong. We bet on which one is right.

## Four Types of Alpha

| Signal | What It Detects | Edge Source |
|---|---|---|
| **Level Divergence** | Sentiment says 70% YES, market says 55% | Crowd has info the market hasn't absorbed |
| **Velocity Divergence** | Sentiment rapidly shifting, price flat | Price will follow sentiment with a lag |
| **Contrarian** | 90% of Reddit agrees (echo chamber) | The crowd is probably wrong; fade them |
| **Cross-Platform** | Reddit says YES, Twitter says NO | One platform has better domain experts |

## Architecture

```
main.py                         CLI: scan, monitor, bet, dashboard, status
├── config.py                   All tunable parameters
├── dashboard.py                Real-time Plotly Dash web dashboard
└── polymarket_predictor/
    ├── polymarket_client.py    Polymarket API (markets, prices, order books)
    ├── reddit_scraper.py       Reddit sentiment via PRAW + JSON fallback
    ├── twitter_scraper.py      X/Twitter sentiment via Tweepy + fallback
    ├── sentiment_analyzer.py   Multi-layer NLP (VADER + TextBlob + keywords)
    ├── echo_chamber.py         Echo chamber detection + narrative momentum
    ├── bayesian_updater.py     Beta-binomial Bayesian probability updating
    ├── divergence_engine.py    Core alpha: sentiment vs. price gap detection
    ├── bankroll_manager.py     Kelly Criterion bet sizing ($50 bankroll)
    └── signal_generator.py     Master orchestrator: all components → signals
```

## Risk Management

- **Quarter-Kelly sizing**: Mathematically optimal but conservative (93% less variance than full Kelly)
- **Max $5 per bet** (10% of bankroll)
- **8% minimum edge** required before any bet is placed
- **Max 5 concurrent bets** — don't spread too thin
- **Echo chamber penalty** — reduces confidence when sentiment is suspiciously uniform
- **Market quality filter** — only bets on liquid, actively-traded markets

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Reddit API credentials (free)
# Twitter Bearer Token is optional but improves signals
```

### Getting API Keys

**Reddit (free, takes 2 minutes)**:
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app"
3. Select "script", give it a name
4. Copy the client ID and secret to your `.env`

**Twitter/X (optional)**:
1. Apply at https://developer.twitter.com
2. Get a Bearer Token
3. Add to `.env`

**Polymarket**: No API key needed for reading markets. Only needed for placing bets programmatically.

## Usage

### Scan for Opportunities
```bash
python main.py scan                # One-time scan of all markets
python main.py scan -v             # Verbose with debug output
```

### Continuous Monitoring
```bash
python main.py monitor             # Re-scans every 30 minutes
python main.py monitor --with-dashboard  # + web dashboard
```

### View/Place Bets
```bash
python main.py bet                 # Show proposed bets
python main.py bet --auto          # Auto-place bets
```

### Dashboard
```bash
python main.py dashboard           # http://localhost:8050
```

### Track Performance
```bash
python main.py status              # Bankroll summary
python main.py history             # Full trade history
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `BANKROLL` | $50.00 | Your total capital |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly for safety |
| `MIN_EDGE_TO_BET` | 8% | Minimum edge before betting |
| `MAX_BET_PCT` | 10% | Max single bet as % of bankroll |
| `MAX_CONCURRENT_BETS` | 5 | Maximum open positions |
| `ECHO_CHAMBER_THRESHOLD` | 85% | When to flag echo chambers |
| `DIVERGENCE_THRESHOLD` | 12% | Minimum sentiment-price gap |
| `SENTIMENT_HALF_LIFE_HOURS` | 12 | How fast old sentiment decays |

## How the Math Works

**Bayesian Updating**: Start with Polymarket price as prior, use sentiment as evidence:
```
Prior:     Beta(α, β) from market price
Evidence:  Pseudocounts from sentiment analysis
Posterior: Beta(α + evidence_yes, β + evidence_no)
Edge:      Posterior - Market Price
```

**Kelly Criterion**: Optimal bet sizing given estimated edge:
```
f* = (bp - q) / b    where b=odds, p=our probability, q=1-p
Bet = Bankroll × f*/4 × confidence    (quarter-Kelly)
```

## Output

- `bankroll.json` — Persisted bankroll state (auto-saved)
- `predictor.log` — Full application log
