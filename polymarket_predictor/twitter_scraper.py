"""
Twitter/X Sentiment Scraper
-----------------------------
Fetches tweets related to Polymarket questions using:
  1. Twitter API v2 (Tweepy) if bearer token is configured
  2. Public search fallback for basic scraping

Extracts tweet text, engagement metrics, and author credibility.
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

import config
from polymarket_predictor.reddit_scraper import SentimentDataPoint

logger = logging.getLogger(__name__)

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.warning("Tweepy not installed. Twitter scraping limited.")


class TwitterScraper:
    """Scrape X/Twitter for sentiment data about Polymarket questions."""

    def __init__(self):
        self._client = None
        if TWEEPY_AVAILABLE and config.TWITTER_BEARER_TOKEN:
            try:
                self._client = tweepy.Client(
                    bearer_token=config.TWITTER_BEARER_TOKEN,
                    wait_on_rate_limit=True,
                )
                logger.info("Twitter API (Tweepy) initialized")
            except Exception as e:
                logger.warning(f"Tweepy init failed: {e}")

    def search_market(
        self,
        market_question: str,
        keywords: Optional[List[str]] = None,
    ) -> List[SentimentDataPoint]:
        """
        Search Twitter for content related to a Polymarket question.
        """
        search_terms = keywords or self._extract_search_terms(market_question)
        if not search_terms:
            return []

        all_points: List[SentimentDataPoint] = []

        for term in search_terms[:2]:  # Twitter rate limits are strict
            if self._client:
                points = self._search_via_api(term)
            else:
                points = self._search_via_fallback(term)
            all_points.extend(points)

        # Deduplicate
        seen = set()
        unique = []
        for p in all_points:
            key = p.text[:100]
            if key not in seen:
                seen.add(key)
                unique.append(p)

        logger.info(
            f"Twitter: found {len(unique)} data points for "
            f"'{market_question[:60]}...'"
        )
        return unique

    def _search_via_api(self, query: str) -> List[SentimentDataPoint]:
        """Search using Twitter API v2."""
        points = []

        try:
            # Add filters: English, no retweets, recent
            full_query = f"{query} lang:en -is:retweet"

            tweets = self._client.search_recent_tweets(
                query=full_query,
                max_results=min(config.TWITTER_SEARCH_LIMIT, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
                user_fields=["verified", "public_metrics"],
                expansions=["author_id"],
            )

            if not tweets or not tweets.data:
                return points

            # Build author lookup
            users = {}
            if tweets.includes and "users" in tweets.includes:
                for user in tweets.includes["users"]:
                    users[user.id] = user

            for tweet in tweets.data:
                text = self._clean_tweet(tweet.text)
                if not text or len(text) < 10:
                    continue

                metrics = tweet.public_metrics or {}
                likes = metrics.get("like_count", 0)
                retweets = metrics.get("retweet_count", 0)
                replies = metrics.get("reply_count", 0)
                engagement = likes + retweets * 2 + replies

                # Author credibility
                author = users.get(tweet.author_id)
                cred = self._compute_credibility(author, engagement)

                timestamp = tweet.created_at or datetime.now(timezone.utc)
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                points.append(SentimentDataPoint(
                    text=text,
                    source="twitter",
                    timestamp=timestamp,
                    score=engagement,
                    author=str(author.username) if author else "",
                    url=f"https://x.com/i/status/{tweet.id}",
                    credibility_weight=cred,
                    metadata={
                        "likes": likes,
                        "retweets": retweets,
                        "replies": replies,
                        "type": "tweet",
                        "verified": getattr(author, "verified", False) if author else False,
                    },
                ))

        except Exception as e:
            logger.error(f"Twitter API search error: {e}")

        return points

    def _search_via_fallback(self, query: str) -> List[SentimentDataPoint]:
        """
        Minimal fallback when no API key is available.
        Uses Nitter-style public search or returns empty.
        """
        points = []

        # Try public search (may be blocked/limited)
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            search_url = (
                f"https://nitter.net/search?f=tweets&q={requests.utils.quote(query)}"
            )
            resp = requests.get(search_url, headers=headers, timeout=10)

            if resp.status_code == 200:
                # Basic text extraction from HTML (very rough)
                from re import findall
                texts = findall(r'class="tweet-content[^"]*"[^>]*>([^<]+)<', resp.text)
                for text in texts[:20]:
                    clean = self._clean_tweet(text.strip())
                    if clean and len(clean) > 15:
                        points.append(SentimentDataPoint(
                            text=clean,
                            source="twitter",
                            timestamp=datetime.now(timezone.utc),
                            score=0,
                            credibility_weight=0.5,  # Low confidence from scraping
                            metadata={"type": "tweet", "method": "fallback"},
                        ))
        except Exception as e:
            logger.debug(f"Twitter fallback search failed: {e}")

        return points

    def _clean_tweet(self, text: str) -> str:
        """Remove URLs, mentions, and excessive whitespace from tweet text."""
        text = re.sub(r"https?://\S+", "", text)           # URLs
        text = re.sub(r"@\w+", "", text)                    # Mentions
        text = re.sub(r"#(\w+)", r"\1", text)               # Keep hashtag text
        text = re.sub(r"\s+", " ", text).strip()            # Whitespace
        return text

    def _extract_search_terms(self, question: str) -> List[str]:
        """Extract search-friendly terms from a Polymarket question."""
        stop_words = {
            "will", "the", "be", "by", "in", "of", "to", "a", "an", "is",
            "are", "was", "were", "has", "have", "do", "does", "this", "that",
            "it", "and", "or", "but", "for", "with", "at", "from", "on",
        }
        clean = re.sub(r"[^\w\s$%]", "", question.lower())
        words = [w for w in clean.split() if w not in stop_words and len(w) > 2]

        terms = []
        if words:
            terms.append(" ".join(words[:4]))
            # Add "polymarket" context
            terms.append(f"polymarket {' '.join(words[:3])}")
        return terms[:3]

    def _compute_credibility(self, author, engagement: int) -> float:
        """Compute credibility weight based on author metrics."""
        if author and getattr(author, "verified", False):
            return config.SOURCE_WEIGHTS["twitter_verified"]

        if author:
            followers = getattr(author, "public_metrics", {}).get("followers_count", 0)
            if followers > 10000:
                return config.SOURCE_WEIGHTS["twitter_verified"]
            elif followers > 1000:
                return config.SOURCE_WEIGHTS["twitter_normal"]

        if engagement > 50:
            return config.SOURCE_WEIGHTS["twitter_normal"]

        return config.SOURCE_WEIGHTS["twitter_low"]
