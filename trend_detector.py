"""
ClipForge — Trend Detector
---------------------------
Pulls real-time trending data from Google Trends and YouTube
for a specific content category in Brazil (or any region).

Used to power:
  1. Trend score in clip scoring (weighted 30%)
  2. Hashtag suggestions per clip

Content categories map to YouTube category IDs and Google Trends categories.

Usage:
  from trend_detector import get_trend_data
  data = get_trend_data("gaming", region="BR")
  print(data["trending_topics"])
  print(data["trending_tags"])
"""

import os
import time
import json
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from pytrends.request import TrendReq
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ─── CATEGORY MAPPINGS ────────────────────────────────────────────────────────

# Maps ClipForge content categories to YouTube category IDs
YOUTUBE_CATEGORIES = {
    "gaming":    "20",   # Gaming
    "tutorial":  "26",   # How-to & Style
    "vlog":      "22",   # People & Blogs
    "podcast":   "22",   # People & Blogs (closest match)
    "comedy":    "23",   # Comedy
    "sports":    "17",   # Sports
    "default":   "0",    # All categories
}

# Maps ClipForge content categories to Google Trends category IDs
# https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
GOOGLE_TRENDS_CATEGORIES = {
    "gaming":    41,     # Computers & Electronics > Video Games
    "tutorial":  958,    # Computers & Electronics > Consumer Electronics
    "vlog":      22,     # Arts & Entertainment
    "podcast":   22,     # Arts & Entertainment
    "comedy":    22,     # Arts & Entertainment
    "sports":    20,     # Sports
    "default":   0,      # All categories
}

# Cache to avoid hitting APIs too frequently
# Trend data doesn't change minute to minute
_cache = {}
CACHE_TTL_MINUTES = 60  # refresh every hour


def _cache_key(content_type: str, region: str) -> str:
    return f"{content_type}_{region}"


def _is_cache_valid(key: str) -> bool:
    if key not in _cache:
        return False
    cached_at = _cache[key].get("cached_at")
    if not cached_at:
        return False
    age = datetime.now() - datetime.fromisoformat(cached_at)
    return age.total_seconds() < CACHE_TTL_MINUTES * 60


# ─── YOUTUBE TRENDING ─────────────────────────────────────────────────────────

def get_youtube_trending(content_type: str, region: str = "BR", max_results: int = 20) -> dict:
    """
    Fetches trending YouTube videos for a specific category and region.
    Returns trending tags, topics, and video titles.
    """
    if not YOUTUBE_API_KEY:
        print("  [Trends] No YouTube API key — skipping YouTube trending")
        return {"tags": [], "topics": [], "titles": []}

    category_id = YOUTUBE_CATEGORIES.get(content_type, YOUTUBE_CATEGORIES["default"])

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        request = youtube.videos().list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=region,
            videoCategoryId=category_id,
            maxResults=max_results,
        )
        response = request.execute()

        tags = []
        topics = []
        titles = []

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            titles.append(snippet.get("title", ""))

            # Collect tags from trending videos
            video_tags = snippet.get("tags", [])
            tags.extend([t.lower() for t in video_tags[:5]])  # top 5 tags per video

            # Collect category/topic info
            topics.append(snippet.get("categoryId", ""))

        # Deduplicate and get most common tags
        from collections import Counter
        tag_counts = Counter(tags)
        top_tags = [tag for tag, count in tag_counts.most_common(20)]

        return {
            "tags": top_tags,
            "topics": list(set(topics)),
            "titles": titles[:10],
        }

    except Exception as e:
        print(f"  [Trends] YouTube trending error: {e}")
        return {"tags": [], "topics": [], "titles": []}


# ─── GOOGLE TRENDS ────────────────────────────────────────────────────────────

def get_google_trends(content_type: str, region: str = "BR", keywords: list = None) -> dict:
    """
    Fetches trending search terms from Google Trends for a category and region.
    If keywords provided, checks their current trend score.
    Returns trending terms and optionally scores for provided keywords.
    """
    try:
        pytrends = TrendReq(hl="pt-BR", tz=180)  # Brazil timezone

        cat_id = GOOGLE_TRENDS_CATEGORIES.get(content_type, 0)

        # Get trending searches in Brazil
        # Map region code to pytrends country name
        region_to_pn = {
            "BR": "brazil",
            "US": "united_states",
            "MX": "mexico",
            "FR": "france",
            "DE": "germany",
            "IT": "italy",
            "JP": "japan",
            "KR": "south_korea",
        }
        pn = region_to_pn.get(region, "united_states")
        try:
            trending_df = pytrends.trending_searches(pn=pn)
            trending_terms = trending_df[0].tolist()[:20] if not trending_df.empty else []
        except Exception:
            # pytrends can fail with 404 for some regions — fall back gracefully
            trending_terms = []

        keyword_scores = {}

        # If we have specific keywords to score, check their interest
        if keywords and len(keywords) > 0:
            # Google Trends allows max 5 keywords per request
            kw_batch = keywords[:5]
            try:
                pytrends.build_payload(
                    kw_batch,
                    cat=cat_id,
                    timeframe="now 7-d",
                    geo=region,
                )
                interest_df = pytrends.interest_over_time()
                if not interest_df.empty:
                    for kw in kw_batch:
                        if kw in interest_df.columns:
                            keyword_scores[kw] = int(interest_df[kw].mean())
            except Exception as e:
                print(f"  [Trends] Keyword scoring error: {e}")

        return {
            "trending_terms": trending_terms,
            "keyword_scores": keyword_scores,
        }

    except Exception as e:
        print(f"  [Trends] Google Trends error: {e}")
        return {"trending_terms": [], "keyword_scores": {}}


# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────

def get_trend_data(content_type: str, region: str = "BR", clip_keywords: list = None) -> dict:
    """
    Main function — combines YouTube trending + Google Trends data.
    Uses caching to avoid repeated API calls.

    Returns:
    {
        "trending_topics": [...],     # What's trending in this category
        "trending_tags": [...],       # Top hashtags from trending videos
        "trending_searches": [...],   # Google Trends terms
        "keyword_scores": {...},      # Trend scores for specific keywords
        "top_titles": [...],          # Titles of trending videos
        "hashtag_suggestions": [...], # Ready-to-use hashtags
    }
    """
    cache_key = _cache_key(content_type, region)

    # Return cached data if still fresh
    if _is_cache_valid(cache_key):
        print(f"  [Trends] Using cached data for {content_type}/{region}")
        cached = _cache[cache_key].copy()
        # Still score clip-specific keywords even with cache
        if clip_keywords:
            fresh_scores = get_google_trends(content_type, region, clip_keywords)
            cached["keyword_scores"] = fresh_scores.get("keyword_scores", {})
        return cached

    print(f"  [Trends] Fetching live trend data for {content_type} in {region}...")

    # Fetch both sources
    yt_data     = get_youtube_trending(content_type, region)
    trends_data = get_google_trends(content_type, region, clip_keywords)

    # Build hashtag suggestions
    # Combine YouTube tags + Google trending terms + category-specific tags
    base_hashtags = _get_base_hashtags(content_type, region)
    yt_hashtags   = ["#" + tag.replace(" ", "").lower() for tag in yt_data["tags"][:8]]
    trend_hashtags = ["#" + term.replace(" ", "").lower() for term in trends_data["trending_terms"][:5]]

    all_hashtags = list(dict.fromkeys(base_hashtags + yt_hashtags + trend_hashtags))[:15]

    result = {
        "content_type":       content_type,
        "region":             region,
        "trending_topics":    yt_data["titles"][:5],
        "trending_tags":      yt_data["tags"][:10],
        "trending_searches":  trends_data["trending_terms"][:10],
        "keyword_scores":     trends_data.get("keyword_scores", {}),
        "top_titles":         yt_data["titles"][:5],
        "hashtag_suggestions": all_hashtags,
        "cached_at":          datetime.now().isoformat(),
    }

    # Cache the result
    _cache[cache_key] = result

    print(f"  [Trends] Found {len(result['trending_tags'])} trending tags, "
          f"{len(result['trending_searches'])} trending searches")

    return result


def _get_base_hashtags(content_type: str, region: str) -> list:
    """
    Returns a base set of always-relevant hashtags for a content type and region.
    These are evergreen tags that perform consistently.
    """
    base = {
        "gaming": [
            "#gaming", "#gamer", "#games", "#gamerbrasil", "#twitchbrasil",
            "#pcgamer", "#streamer", "#streamerbrasil", "#gameplay",
        ],
        "tutorial": [
            "#tutorial", "#dicas", "#dicasgamer", "#setupgamer", "#pcgamer",
            "#tecnologia", "#setup", "#howto",
        ],
        "vlog": [
            "#vlog", "#vlogbrasil", "#lifestyle", "#brasil", "#rotina",
            "#diavlog", "#vlogdiario",
        ],
        "podcast": [
            "#podcast", "#podcastbrasil", "#podcastpt", "#entrevista",
            "#conversa", "#debate",
        ],
        "comedy": [
            "#humor", "#comedia", "#engraçado", "#meme", "#viral",
            "#brazilianhumor", "#risadas",
        ],
        "sports": [
            "#futebol", "#soccer", "#brasil", "#gol", "#golaço",
            "#futebolbrasileiro", "#seleção", "#highlight",
            "#football", "#sportsbr",
        ],
    }

    hashtags = base.get(content_type, ["#viral", "#brasil", "#trending"])

    # Add region-specific tags
    if region == "BR":
        hashtags += ["#brasil", "#brazil", "#brasileiro"]

    return hashtags


def calculate_trend_score(clip_topics: list, trend_data: dict) -> int:
    """
    Calculates a trend alignment score (0-100) for a clip
    based on how well its content matches current trending data.

    clip_topics: keywords/topics extracted from the clip by Gemini
    trend_data: output from get_trend_data()
    """
    if not clip_topics or not trend_data:
        return 50  # neutral score if no data

    trending_tags     = [t.lower() for t in trend_data.get("trending_tags", [])]
    trending_searches = [t.lower() for t in trend_data.get("trending_searches", [])]
    trending_titles   = " ".join(trend_data.get("top_titles", [])).lower()

    score = 50  # start neutral
    matches = 0

    for topic in clip_topics:
        topic_lower = topic.lower()

        # Check YouTube trending tags
        if any(topic_lower in tag or tag in topic_lower for tag in trending_tags):
            score += 8
            matches += 1

        # Check Google trending searches
        if any(topic_lower in search or search in topic_lower for search in trending_searches):
            score += 10
            matches += 1

        # Check trending video titles
        if topic_lower in trending_titles:
            score += 5
            matches += 1

    # Bonus for multiple matches
    if matches >= 3:
        score += 10
    elif matches >= 2:
        score += 5

    return min(100, max(0, score))


if __name__ == "__main__":
    # Test it
    print("\nTesting trend detector...\n")
    for category in ["gaming", "podcast", "tutorial"]:
        print(f"\n{'─' * 40}")
        print(f"Category: {category.upper()}")
        data = get_trend_data(category, region="BR")
        print(f"Trending tags    : {data['trending_tags'][:5]}")
        print(f"Trending searches: {data['trending_searches'][:5]}")
        print(f"Hashtags         : {data['hashtag_suggestions'][:8]}")
