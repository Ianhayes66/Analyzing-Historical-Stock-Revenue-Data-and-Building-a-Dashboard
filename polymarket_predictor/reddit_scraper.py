"""
Reddit Sentiment Scraper
-------------------------
Scrapes Reddit for posts and comments related to Polymarket questions.
Uses PRAW (Python Reddit API Wrapper) for the official API, with a
fallback to public JSON endpoints for when API keys aren't configured.

The scraper:
  1. Searches relevant subreddits for market-related keywords
  2. Extracts post titles, bodies, and top comments
  3. Tags each piece of text with metadata (karma, age, subreddit)
  4. Returns structured SentimentDataPoint objects for the NLP pipeline
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import requests

import config

logger = logging.getLogger(__name__)

# Try to import PRAW — graceful fallback if not configured
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("PRAW not installed. Using public JSON fallback.")


@dataclass
class SentimentDataPoint:
    """A single piece of text with metadata for sentiment analysis."""
    text: str
    source: str               # "reddit" or "twitter"
    timestamp: datetime
    score: int = 0            # Upvotes/likes
    author: str = ""
    subreddit: str = ""
    url: str = ""
    credibility_weight: float = 1.0  # Based on karma, engagement
    metadata: dict = field(default_factory=dict)


class RedditScraper:
    """Scrape Reddit for sentiment data about Polymarket questions."""

    def __init__(self):
        self._reddit = None
        if PRAW_AVAILABLE and config.REDDIT_CLIENT_ID:
            try:
                self._reddit = praw.Reddit(
                    client_id=config.REDDIT_CLIENT_ID,
                    client_secret=config.REDDIT_CLIENT_SECRET,
                    user_agent=config.REDDIT_USER_AGENT,
                )
                logger.info("Reddit API (PRAW) initialized")
            except Exception as e:
                logger.warning(f"PRAW init failed: {e}. Using JSON fallback.")

    def search_market(
        self,
        market_question: str,
        keywords: Optional[List[str]] = None,
    ) -> List[SentimentDataPoint]:
        """
        Search Reddit for content related to a Polymarket question.
        Extracts keywords from the question and searches across subreddits.
        """
        search_terms = keywords or self._extract_search_terms(market_question)
        if not search_terms:
            return []

        all_points: List[SentimentDataPoint] = []

        for term in search_terms[:3]:  # Limit to top 3 terms
            if self._reddit:
                points = self._search_via_praw(term)
            else:
                points = self._search_via_json(term)
            all_points.extend(points)

        # Deduplicate by URL
        seen = set()
        unique = []
        for p in all_points:
            key = p.url or p.text[:80]
            if key not in seen:
                seen.add(key)
                unique.append(p)

        logger.info(
            f"Reddit: found {len(unique)} data points for "
            f"'{market_question[:60]}...'"
        )
        return unique

    def _search_via_praw(self, query: str) -> List[SentimentDataPoint]:
        """Search using the official Reddit API."""
        points = []

        for sub_name in config.REDDIT_SUBREDDITS:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                results = subreddit.search(
                    query, sort="relevance", time_filter="week",
                    limit=config.REDDIT_POST_LIMIT,
                )

                for post in results:
                    # Post itself
                    text = f"{post.title}. {post.selftext}" if post.selftext else post.title
                    cred = self._compute_credibility(post.score, sub_name)

                    points.append(SentimentDataPoint(
                        text=text[:2000],
                        source="reddit",
                        timestamp=datetime.fromtimestamp(
                            post.created_utc, tz=timezone.utc
                        ),
                        score=post.score,
                        author=str(post.author) if post.author else "",
                        subreddit=sub_name,
                        url=f"https://reddit.com{post.permalink}",
                        credibility_weight=cred,
                        metadata={
                            "num_comments": post.num_comments,
                            "upvote_ratio": post.upvote_ratio,
                            "type": "post",
                        },
                    ))

                    # Top comments
                    try:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments[:config.REDDIT_COMMENT_DEPTH]:
                            if not comment.body or comment.body == "[deleted]":
                                continue
                            c_cred = self._compute_credibility(comment.score, sub_name)
                            points.append(SentimentDataPoint(
                                text=comment.body[:1000],
                                source="reddit",
                                timestamp=datetime.fromtimestamp(
                                    comment.created_utc, tz=timezone.utc
                                ),
                                score=comment.score,
                                author=str(comment.author) if comment.author else "",
                                subreddit=sub_name,
                                url=f"https://reddit.com{comment.permalink}",
                                credibility_weight=c_cred,
                                metadata={"type": "comment"},
                            ))
                    except Exception:
                        pass  # Comments sometimes fail — non-critical

            except Exception as e:
                logger.debug(f"Error searching r/{sub_name} for '{query}': {e}")

        return points

    def _search_via_json(self, query: str) -> List[SentimentDataPoint]:
        """
        Fallback: search Reddit's public JSON endpoints.
        No API key needed but rate-limited.
        """
        points = []
        headers = {"User-Agent": config.REDDIT_USER_AGENT}

        for sub_name in config.REDDIT_SUBREDDITS[:5]:  # Fewer subs in fallback
            try:
                url = (
                    f"https://www.reddit.com/r/{sub_name}/search.json"
                    f"?q={requests.utils.quote(query)}"
                    f"&sort=relevance&t=week&limit=25&restrict_sr=on"
                )
                resp = requests.get(url, headers=headers, timeout=10)

                if resp.status_code == 429:
                    logger.warning("Reddit rate limited. Sleeping 5s.")
                    time.sleep(5)
                    continue

                resp.raise_for_status()
                data = resp.json()

                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    title = post.get("title", "")
                    body = post.get("selftext", "")
                    text = f"{title}. {body}" if body else title

                    if not text.strip():
                        continue

                    created = post.get("created_utc", 0)
                    score = post.get("score", 0)
                    cred = self._compute_credibility(score, sub_name)

                    points.append(SentimentDataPoint(
                        text=text[:2000],
                        source="reddit",
                        timestamp=datetime.fromtimestamp(created, tz=timezone.utc),
                        score=score,
                        author=post.get("author", ""),
                        subreddit=sub_name,
                        url=f"https://reddit.com{post.get('permalink', '')}",
                        credibility_weight=cred,
                        metadata={
                            "num_comments": post.get("num_comments", 0),
                            "upvote_ratio": post.get("upvote_ratio", 0.5),
                            "type": "post",
                        },
                    ))

                time.sleep(1)  # Be polite to the API

            except Exception as e:
                logger.debug(f"JSON fallback error for r/{sub_name}: {e}")

        return points

    def _extract_search_terms(self, question: str) -> List[str]:
        """
        Extract meaningful search terms from a market question.
        E.g., "Will Bitcoin reach $100k by March 2025?" -> ["Bitcoin $100k", "Bitcoin price"]
        """
        # Remove common filler words
        stop_words = {
            "will", "the", "be", "by", "in", "of", "to", "a", "an", "is",
            "are", "was", "were", "has", "have", "had", "do", "does", "did",
            "this", "that", "these", "those", "it", "its", "and", "or", "but",
            "for", "with", "at", "from", "on", "as", "if", "than", "more",
            "before", "after", "above", "below", "between", "during",
        }

        # Clean the question
        clean = re.sub(r"[^\w\s$%]", "", question.lower())
        words = [w for w in clean.split() if w not in stop_words and len(w) > 2]

        if not words:
            return [question[:50]]

        # Build search queries
        terms = []

        # Full key phrase (first ~5 meaningful words)
        terms.append(" ".join(words[:5]))

        # Pairs of important words
        if len(words) >= 2:
            terms.append(f"{words[0]} {words[1]}")

        # Individual important words (proper nouns, numbers)
        for w in words:
            if w[0].isupper() or w.startswith("$") or w.isdigit():
                terms.append(w)

        return terms[:4]

    def _compute_credibility(self, score: int, subreddit: str) -> float:
        """
        Assign credibility weight based on engagement.
        High-karma posts carry more weight.
        """
        if score > 100:
            base = config.SOURCE_WEIGHTS["reddit_high_karma"]
        elif score < 0:
            base = config.SOURCE_WEIGHTS["reddit_low_karma"]
        else:
            base = config.SOURCE_WEIGHTS["reddit_normal"]

        # Boost prediction-focused subreddits
        prediction_subs = {"polymarket", "predictor", "wallstreetbets"}
        if subreddit.lower() in prediction_subs:
            base *= 1.3

        return base
