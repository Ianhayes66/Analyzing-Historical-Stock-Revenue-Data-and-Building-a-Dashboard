"""
Configuration — Polymarket Sentiment Predictor
================================================
All tunable parameters in one place. Copy .env.example to .env
and fill in your API keys.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Bankroll ────────────────────────────────────────────────────────────
BANKROLL = 50.00                    # Your total capital ($50)
MAX_BET_PCT = 0.10                  # Max 10% of bankroll per bet ($5)
KELLY_FRACTION = 0.25               # Quarter-Kelly — very conservative
MIN_EDGE_TO_BET = 0.08              # Need 8%+ estimated edge to bet
MIN_BET_SIZE = 0.50                 # Minimum bet (Polymarket minimum)
MAX_CONCURRENT_BETS = 5             # Don't spread too thin
BANKROLL_FILE = "bankroll.json"

# ── Polymarket ──────────────────────────────────────────────────────────
POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
# Only needed if you want to actually place bets (leave blank for analysis-only)
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_SECRET = os.getenv("POLYMARKET_SECRET", "")

# Market filters
MIN_MARKET_LIQUIDITY = 5000         # Ignore illiquid markets ($ volume)
MIN_MARKET_VOLUME = 1000            # Minimum 24h volume
MAX_DAYS_TO_RESOLUTION = 30         # Prefer near-term markets (faster feedback)
MIN_DAYS_TO_RESOLUTION = 0.25       # Avoid last-minute markets (6 hours)

# ── Reddit ──────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "PolymarketPredictor/1.0"

# Subreddits to monitor (ordered by signal quality)
REDDIT_SUBREDDITS = [
    "polymarket",
    "Predictor",
    "politics",
    "news",
    "worldnews",
    "economics",
    "wallstreetbets",
    "sports",
    "nfl",
    "nba",
    "soccer",
    "cryptocurrency",
    "technology",
]

REDDIT_POST_LIMIT = 100             # Posts per subreddit per scan
REDDIT_COMMENT_DEPTH = 50           # Comments to analyze per relevant post

# ── Twitter/X ───────────────────────────────────────────────────────────
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
TWITTER_SEARCH_LIMIT = 200          # Tweets per search query

# ── Sentiment Analysis ──────────────────────────────────────────────────
# Weights for combining VADER, TextBlob, and keyword sentiment
SENTIMENT_WEIGHTS = {
    "vader": 0.45,
    "textblob": 0.25,
    "keyword": 0.30,
}

# Time decay: recent posts matter exponentially more
SENTIMENT_HALF_LIFE_HOURS = 12      # Sentiment weight halves every 12 hours

# Source credibility weights
SOURCE_WEIGHTS = {
    "reddit_high_karma": 1.5,       # Posts with >100 upvotes
    "reddit_normal": 1.0,
    "reddit_low_karma": 0.5,        # Posts with <0 score
    "twitter_verified": 1.8,        # Verified/high-follower accounts
    "twitter_normal": 1.0,
    "twitter_low": 0.4,             # Low-follower accounts
}

# ── Echo Chamber Detection ──────────────────────────────────────────────
ECHO_CHAMBER_THRESHOLD = 0.85       # If >85% sentiment agrees, flag it
ECHO_CHAMBER_PENALTY = 0.40         # Reduce signal confidence by 40%
CONTRARIAN_BOOST = 1.25             # Boost contrarian signals by 25%
MIN_SAMPLE_SIZE = 15                # Need at least 15 data points

# ── Bayesian Updater ────────────────────────────────────────────────────
PRIOR_WEIGHT = 0.60                 # How much to trust market price as prior
EVIDENCE_WEIGHT = 0.40              # How much sentiment can shift the estimate
BAYESIAN_CONFIDENCE_FLOOR = 0.05    # Never go below 5% or above 95%
BAYESIAN_CONFIDENCE_CEILING = 0.95

# ── Divergence Engine ───────────────────────────────────────────────────
# The core alpha — when sentiment and price disagree
DIVERGENCE_THRESHOLD = 0.12         # 12%+ gap between sentiment prob and market price
DIVERGENCE_LOOKBACK_HOURS = 48      # How far back to analyze sentiment trend
VELOCITY_WINDOW_HOURS = 6           # Sentiment velocity measurement window

# ── Signal Scoring ──────────────────────────────────────────────────────
SIGNAL_COMPONENTS = {
    "divergence":       0.35,       # Sentiment vs. price gap (primary signal)
    "velocity":         0.20,       # How fast sentiment is shifting
    "echo_chamber":     0.15,       # Contrarian / echo chamber adjustment
    "bayesian":         0.15,       # Bayesian probability estimate
    "volume_sentiment": 0.15,       # Are volume and sentiment aligned?
}

# ── Dashboard ───────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
REFRESH_INTERVAL_SEC = 60           # Dashboard auto-refresh

# ── Scan Schedule ───────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 30          # How often to re-scan everything
