"""
ClipForge — Chat Spike Detector
--------------------------------
Pulls YouTube video comments and/or live chat replay,
detects reaction spikes, and returns ranked timestamps.

These timestamps feed directly into the Gemini pipeline
as a second signal for finding viral clip moments.

How it works:
  1. Fetch all comments/chat messages with timestamps
  2. Bucket messages into 10-second windows
  3. Score each window by reaction word density + message volume
  4. Flag windows that spike above the baseline
  5. Return ranked spike timestamps ready to cross-reference with Gemini

Usage:
  python chat_spike_detector.py <youtube_url_or_video_id>
"""

import os
import re
import json
import sys
from collections import defaultdict
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ─── REACTION WORD LISTS ──────────────────────────────────────────────────────
# Words that signal a viral/funny/hype moment in chat
# Weighted by how strongly they indicate a reaction

REACTION_WORDS = {
    # Tier 1 — strongest signals (weight: 3)
    "high": [
        "lmaooo", "lmaoooo", "lmaooooo", "omfg", "wtfffff", "noooo",
        "yoooo", "yooooo", "lets gooo", "letsgooo", "lets goooo",
        "bro what", "no way", "no wayyy", "clip it", "clip that",
        "clipclipclip", "clip!", "timestamp", "w moment", "greatest",
        "insane", "clip this", "holy shit", "holy moly",
        # Portuguese equivalents
        "kkkkkk", "kkkkkkk", "kkkkkkkk", "mds", "cara", "que isso",
        "nossa", "mano", "que", "caralho", "clipa", "clipa isso",
    ],
    # Tier 2 — strong signals (weight: 2)
    "medium": [
        "lmao", "lmfao", "omg", "oof", "rip", "gg", "ez",
        "lets go", "pog", "poggers", "based", "w", "l", "ratio",
        "dead", "im dead", "💀", "😭", "😂", "🤣", "💯", "🔥",
        "dude", "bro", "bruh", "sheesh", "goat", "god", "crazy",
        # Portuguese
        "kkk", "kkkk", "kkkkk", "hahaha", "aaaaa", "uau",
        "que loucura", "absurdo", "mito",
    ],
    # Tier 3 — moderate signals (weight: 1)
    "low": [
        "lol", "haha", "hahaha", "wow", "nice", "damn", "yep",
        "yes", "no", "what", "wait", "oh", "ah", "ok", "okay",
        "👀", "😮", "🎉", "❤️", "🤯",
        # Portuguese
        "que", "isso", "sim", "nao", "oi",
    ],
}

# Flatten into lookup dict: word -> weight
WORD_WEIGHTS = {}
for weight, (tier, words) in enumerate(
    [("low", 1), ("medium", 2), ("high", 3)], start=1
):
    pass

WORD_WEIGHTS = {}
for tier, words in REACTION_WORDS.items():
    w = {"high": 3, "medium": 2, "low": 1}[tier]
    for word in words:
        WORD_WEIGHTS[word.lower()] = w

# Timestamp pattern in comments e.g. "2:14" or "1:02:33"
TIMESTAMP_PATTERN = re.compile(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b')


def extract_video_id(url_or_id: str) -> str:
    """Extracts YouTube video ID from a URL or returns the ID directly."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def timestamp_to_seconds(h_or_m, m_or_s, s=None) -> int:
    """Converts timestamp parts to total seconds."""
    if s is not None:
        return int(h_or_m) * 3600 + int(m_or_s) * 60 + int(s)
    return int(h_or_m) * 60 + int(m_or_s)


def score_message(text: str) -> int:
    """
    Scores a single chat message by reaction word content.
    Returns a score 0-10+
    """
    text_lower = text.lower()
    score = 0

    # Check for reaction words
    for word, weight in WORD_WEIGHTS.items():
        if word in text_lower:
            score += weight

    # Bonus for ALL CAPS (shouting = more excited)
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    score += len(caps_words) * 0.5

    # Bonus for repeated characters (LMAOOOO vs LMAO)
    if re.search(r'(.)\1{3,}', text):
        score += 1

    # Bonus for multiple exclamation marks
    exclamations = text.count('!')
    if exclamations >= 3:
        score += 1

    return round(score, 1)


def extract_comment_timestamps(comments: list) -> list:
    """
    Finds user-written timestamps in comments like "2:14 best moment".
    These are crowdsourced clip suggestions — pure gold.
    """
    found = []
    for comment in comments:
        text = comment.get("text", "")
        matches = TIMESTAMP_PATTERN.finditer(text)
        for match in matches:
            parts = [g for g in match.groups() if g is not None]
            if len(parts) == 2:
                secs = timestamp_to_seconds(parts[0], parts[1])
            else:
                secs = timestamp_to_seconds(parts[0], parts[1], parts[2])

            # Get surrounding context (the comment text near the timestamp)
            context = text[max(0, match.start()-20):match.end()+60].strip()
            found.append({
                "timestamp_seconds": secs,
                "timestamp_display": match.group(0),
                "context": context,
                "likes": comment.get("likes", 0),
            })

    return found


def fetch_video_comments(video_id: str, max_comments: int = 500) -> list:
    """
    Fetches top-level comments from a YouTube video.
    Sorted by relevance (most liked first) to get the best signals.
    """
    print(f"  Fetching video comments (up to {max_comments})...")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments = []
    next_page = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            order="relevance",
            pageToken=next_page,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            })

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    print(f"  Fetched {len(comments)} comments")
    return comments


def fetch_live_chat(video_id: str, max_messages: int = 2000) -> list:
    """
    Fetches live chat messages from a YouTube livestream VOD.
    These include offsetTimeMsec — the exact timestamp in the stream.
    Only works for videos that were live streams.
    """
    print(f"  Fetching live chat replay (up to {max_messages} messages)...")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # First get the live chat ID from the video
    video_response = youtube.videos().list(
        part="liveStreamingDetails",
        id=video_id,
    ).execute()

    items = video_response.get("items", [])
    if not items:
        print("  No live streaming details found — video may not be a livestream")
        return []

    live_details = items[0].get("liveStreamingDetails", {})
    chat_id = live_details.get("activeLiveChatId")

    if not chat_id:
        print("  No live chat ID found — chat replay may not be available")
        return []

    print(f"  Live chat ID found: {chat_id[:20]}...")

    messages = []
    next_page = None

    while len(messages) < max_messages:
        request = youtube.liveChatMessages().list(
            liveChatId=chat_id,
            part="snippet,authorDetails",
            maxResults=min(200, max_messages - len(messages)),
            pageToken=next_page,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            msg_type = snippet.get("type", "")

            if msg_type == "textMessageEvent":
                offset_ms = snippet.get("videoOffsetTimeMsec")
                text = snippet.get("displayMessage", "")

                if offset_ms is not None:
                    messages.append({
                        "text": text,
                        "timestamp_seconds": int(offset_ms) // 1000,
                        "type": "chat",
                        "author": item.get("authorDetails", {}).get("displayName", ""),
                    })

            elif msg_type == "superChatEvent":
                # Super Chats = someone paid = big moment
                offset_ms = snippet.get("videoOffsetTimeMsec")
                amount = snippet.get("superChatDetails", {}).get("amountDisplayString", "")
                if offset_ms is not None:
                    messages.append({
                        "text": f"[SUPER CHAT {amount}] {snippet.get('displayMessage', '')}",
                        "timestamp_seconds": int(offset_ms) // 1000,
                        "type": "superchat",
                        "author": item.get("authorDetails", {}).get("displayName", ""),
                    })

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    print(f"  Fetched {len(messages)} chat messages")
    return messages


def detect_spikes(
    messages: list,
    window_seconds: int = 10,
    spike_multiplier: float = 2.5,
) -> list:
    """
    Buckets messages into time windows and detects spikes.

    A spike = a window where the reaction score is significantly
    higher than the baseline average. That's your clip moment.

    Returns list of spike timestamps sorted by score descending.
    """
    if not messages:
        return []

    # Build time-bucketed scoring
    buckets = defaultdict(lambda: {"score": 0, "count": 0, "messages": []})

    for msg in messages:
        ts = msg.get("timestamp_seconds", 0)
        bucket_key = (ts // window_seconds) * window_seconds  # snap to window
        msg_score = score_message(msg["text"])

        # Super Chats automatically get a big score boost
        if msg.get("type") == "superchat":
            msg_score += 10

        buckets[bucket_key]["score"] += msg_score
        buckets[bucket_key]["count"] += 1
        buckets[bucket_key]["messages"].append(msg["text"][:80])

    if not buckets:
        return []

    # Calculate baseline (median score across all windows)
    all_scores = [b["score"] for b in buckets.values()]
    all_scores.sort()
    median_score = all_scores[len(all_scores) // 2]
    baseline = max(median_score, 1)  # avoid division by zero

    # Find spikes
    spikes = []
    for timestamp, data in buckets.items():
        spike_ratio = data["score"] / baseline

        if spike_ratio >= spike_multiplier:
            # Format timestamp as MM:SS
            mins = timestamp // 60
            secs = timestamp % 60
            display = f"{mins}:{secs:02d}"

            spikes.append({
                "timestamp_seconds": timestamp,
                "timestamp_display": display,
                "spike_ratio": round(spike_ratio, 1),
                "reaction_score": round(data["score"], 1),
                "message_count": data["count"],
                "sample_messages": data["messages"][:5],
                "confidence": min(100, int(spike_ratio * 20)),
            })

    # Sort by spike ratio descending
    spikes.sort(key=lambda x: x["spike_ratio"], reverse=True)
    return spikes


def format_seconds(s: int) -> str:
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def run_spike_detector(url_or_id: str, is_livestream: bool = False) -> dict:
    """
    Full spike detection pipeline.
    Returns combined results from comment timestamps + chat spikes.
    """
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY not found. Add it to your .env file.")

    video_id = extract_video_id(url_or_id)
    print(f"\n[Chat Spike Detector] Video ID: {video_id}")
    print("─" * 50)

    results = {
        "video_id": video_id,
        "chat_spikes": [],
        "comment_timestamps": [],
        "combined_moments": [],
    }

    # ── LIVE CHAT (if livestream VOD) ──────────────────────────────────────
    if is_livestream:
        print("\n[1/3] Pulling live chat replay...")
        chat_messages = fetch_live_chat(video_id)

        if chat_messages:
            print("\n[2/3] Detecting reaction spikes in chat...")
            spikes = detect_spikes(chat_messages)
            results["chat_spikes"] = spikes
            print(f"  Found {len(spikes)} spikes")

            # Print spike summary
            print("\n  Top chat spikes:")
            for spike in spikes[:5]:
                bar = "█" * min(20, int(spike["spike_ratio"] * 3))
                print(f"  {spike['timestamp_display']:>8}  [{bar:<20}] "
                      f"x{spike['spike_ratio']} | "
                      f"{spike['message_count']} msgs | "
                      f"confidence {spike['confidence']}%")
                for msg in spike["sample_messages"][:2]:
                    print(f"             → \"{msg[:60]}\"")
        else:
            print("  No live chat data available")

    # ── VOD COMMENTS (all videos) ──────────────────────────────────────────
    step = "[3/3]" if is_livestream else "[1/2]"
    print(f"\n{step} Fetching video comments for user timestamps...")
    comments = fetch_video_comments(video_id)

    step2 = "[3/3]" if not is_livestream else ""
    if not is_livestream:
        print(f"\n[2/2] Scanning comments for user-written timestamps...")

    comment_timestamps = extract_comment_timestamps(comments)
    results["comment_timestamps"] = comment_timestamps

    if comment_timestamps:
        # Sort by likes (most upvoted timestamp comments = most agreed upon)
        comment_timestamps.sort(key=lambda x: x["likes"], reverse=True)
        print(f"  Found {len(comment_timestamps)} user-written timestamps")
        print("\n  Top community timestamps:")
        for ct in comment_timestamps[:5]:
            print(f"  {ct['timestamp_display']:>8}  "
                  f"({ct['likes']} likes) → \"{ct['context'][:60]}\"")
    else:
        print("  No user timestamps found in comments")

    # ── COMBINE ALL SIGNALS ────────────────────────────────────────────────
    print("\n─" * 50)
    print("Combining all signals into moment rankings...")

    moment_map = defaultdict(lambda: {
        "timestamp_seconds": 0,
        "timestamp_display": "",
        "total_score": 0,
        "signals": [],
    })

    # Add chat spikes
    for spike in results["chat_spikes"]:
        ts = spike["timestamp_seconds"]
        moment_map[ts]["timestamp_seconds"] = ts
        moment_map[ts]["timestamp_display"] = spike["timestamp_display"]
        moment_map[ts]["total_score"] += spike["confidence"]
        moment_map[ts]["signals"].append(f"Chat spike x{spike['spike_ratio']} ({spike['message_count']} msgs)")

    # Add comment timestamps (nearby windows get combined)
    for ct in results["comment_timestamps"]:
        # Snap to nearest 10-second bucket
        ts = (ct["timestamp_seconds"] // 10) * 10
        display = format_seconds(ts)
        likes_score = min(50, ct["likes"] * 2)  # cap at 50
        moment_map[ts]["timestamp_seconds"] = ts
        moment_map[ts]["timestamp_display"] = display
        moment_map[ts]["total_score"] += likes_score + 20  # base 20 for any mention
        moment_map[ts]["signals"].append(
            f"Community timestamp ({ct['likes']} likes): \"{ct['context'][:40]}\""
        )

    # Sort combined moments by total score
    combined = sorted(moment_map.values(), key=lambda x: x["total_score"], reverse=True)
    results["combined_moments"] = combined

    # Final summary
    print(f"\n{'═' * 50}")
    print(f"TOP MOMENTS FROM AUDIENCE SIGNALS:")
    print(f"{'═' * 50}")
    for i, moment in enumerate(combined[:8], 1):
        print(f"\n  #{i} — {moment['timestamp_display']} (score: {moment['total_score']})")
        for signal in moment["signals"]:
            print(f"       • {signal}")

    print(f"\n{'─' * 50}")
    print(f"Ready to cross-reference with Gemini audio analysis.")
    print(f"{'─' * 50}\n")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python chat_spike_detector.py <youtube_url> [--live]")
        print("  --live : include live chat replay (for livestream VODs)")
        print("\nExamples:")
        print('  python chat_spike_detector.py "https://youtube.com/watch?v=xyz"')
        print('  python chat_spike_detector.py "https://youtube.com/watch?v=xyz" --live')
        sys.exit(1)

    url = sys.argv[1]
    is_live = "--live" in sys.argv

    results = run_spike_detector(url, is_livestream=is_live)

    output_file = "chat_spikes.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}\n")
