"""
ClipForge — Signal Combiner
----------------------------
Merges two independent signals into one ranked clip list:

  Signal A: Gemini Flash audio analysis (energy, tone, emotion)
  Signal B: Chat spike detector (audience reaction, timestamps, comments)

A clip that scores high on BOTH signals = highest confidence viral moment.
A clip that only scores on one = still worth considering, lower confidence.

This is ClipForge's core differentiator. No competitor does this.

Usage:
  from combiner import combine_signals
  final_clips = combine_signals(gemini_results, chat_results, video_duration)
"""

import json
from typing import Optional


def seconds_to_display(seconds: int) -> str:
    """Convert seconds to MM:SS or HH:MM:SS display format."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def display_to_seconds(display: str) -> int:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = display.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def timestamps_are_close(ts1: int, ts2: int, tolerance_seconds: int = 20) -> bool:
    """
    Returns True if two timestamps are within tolerance of each other.
    Used to match Gemini clips to chat spikes that happen nearby.
    """
    return abs(ts1 - ts2) <= tolerance_seconds


def combine_signals(
    gemini_results: dict,
    chat_results: dict,
    clip_length: int = 45,
    max_clips: int = 5,
) -> list:
    """
    Combines Gemini audio analysis with chat spike data.

    Scoring system:
    ─────────────────────────────────────────────────
    Gemini viral score (0-100)        → up to 50 pts
    Chat spike confidence (0-100)     → up to 30 pts
    Community timestamp present       → +15 pts
    Super Chat at this moment         → +10 pts
    Both signals agree (within 20s)   → +10 pts bonus
    ─────────────────────────────────────────────────
    Max possible score: 115 pts
    """

    gemini_clips = gemini_results.get("clips", [])
    chat_moments = chat_results.get("combined_moments", [])
    chat_spikes = chat_results.get("chat_spikes", [])
    comment_timestamps = chat_results.get("comment_timestamps", [])

    final_clips = []

    for clip in gemini_clips:
        # Parse Gemini timestamps
        start_secs = display_to_seconds(clip.get("start_time", "0:00"))
        end_secs = display_to_seconds(clip.get("end_time", "0:45"))
        gemini_score = clip.get("viral_score", 50)

        # Base score from Gemini (scaled to 50 pts max)
        base_score = (gemini_score / 100) * 50

        # Build signal tracking
        signals_found = []
        bonus_score = 0
        chat_confidence = 0
        has_community_ts = False
        has_superchat = False
        chat_signal_detail = None

        # ── Check for nearby chat spikes ───────────────────────────────────
        for spike in chat_spikes:
            spike_ts = spike.get("timestamp_seconds", 0)
            if timestamps_are_close(start_secs, spike_ts):
                chat_confidence = max(chat_confidence, spike.get("confidence", 0))
                chat_signal_detail = spike
                signals_found.append(
                    f"Chat spike x{spike.get('spike_ratio', 0)} "
                    f"at {spike.get('timestamp_display')} "
                    f"({spike.get('message_count', 0)} msgs)"
                )

                # Check if any messages in spike are Super Chats
                sample = " ".join(spike.get("sample_messages", []))
                if "SUPER CHAT" in sample.upper():
                    has_superchat = True

        # ── Check for community timestamps ─────────────────────────────────
        for ct in comment_timestamps:
            ct_secs = ct.get("timestamp_seconds", 0)
            if timestamps_are_close(start_secs, ct_secs, tolerance_seconds=30):
                has_community_ts = True
                likes = ct.get("likes", 0)
                signals_found.append(
                    f"Community timestamp ({likes} likes): "
                    f"\"{ct.get('context', '')[:50]}\""
                )

        # ── Check combined moment map ───────────────────────────────────────
        for moment in chat_moments:
            moment_ts = moment.get("timestamp_seconds", 0)
            if timestamps_are_close(start_secs, moment_ts):
                bonus_score += 5  # general agreement bonus

        # ── Apply scoring ──────────────────────────────────────────────────
        chat_score = (chat_confidence / 100) * 30
        community_bonus = 15 if has_community_ts else 0
        superchat_bonus = 10 if has_superchat else 0

        # Big bonus when BOTH Gemini and chat agree on the same moment
        both_agree_bonus = 10 if (chat_confidence > 0 and gemini_score >= 70) else 0

        total_score = (
            base_score
            + chat_score
            + community_bonus
            + superchat_bonus
            + both_agree_bonus
            + bonus_score
        )

        # Determine confidence tier
        if total_score >= 80:
            confidence_tier = "🔥 Very High"
        elif total_score >= 60:
            confidence_tier = "⚡ High"
        elif total_score >= 40:
            confidence_tier = "✅ Medium"
        else:
            confidence_tier = "📊 Low"

        final_clips.append({
            "title": clip.get("title", "Untitled clip"),
            "start_time": clip.get("start_time"),
            "end_time": clip.get("end_time"),
            "start_seconds": start_secs,
            "end_seconds": end_secs,
            "duration_seconds": end_secs - start_secs or clip_length,
            "total_score": round(total_score, 1),
            "confidence_tier": confidence_tier,

            # Individual signal scores
            "gemini_score": gemini_score,
            "chat_confidence": chat_confidence,
            "has_community_timestamp": has_community_ts,
            "has_superchat": has_superchat,

            # Content for the clip
            "suggested_caption": clip.get("suggested_caption", ""),
            "hook_line": clip.get("hook_line", ""),
            "clip_type": clip.get("clip_type", "viral"),
            "energy_level": clip.get("energy_level", "medium"),
            "reason": clip.get("reason", ""),

            # What signals contributed
            "signals": signals_found if signals_found else ["Gemini audio analysis only"],
            "both_signals": len(signals_found) > 0,
        })

    # ── Also surface strong chat spikes Gemini didn't catch ───────────────
    # Sometimes the audio isn't remarkable but the chat went INSANE
    # (stream crashes, off-camera moments, inside jokes, etc.)
    covered_timestamps = {c["start_seconds"] for c in final_clips}

    for spike in chat_spikes:
        spike_ts = spike.get("timestamp_seconds", 0)

        # Skip if already covered by a Gemini clip
        already_covered = any(
            timestamps_are_close(spike_ts, ts, 30)
            for ts in covered_timestamps
        )
        if already_covered:
            continue

        # Only surface very strong spikes
        if spike.get("confidence", 0) >= 70:
            chat_only_score = (spike["confidence"] / 100) * 30 + 10
            if spike.get("type") == "superchat":
                chat_only_score += 10

            end_ts = spike_ts + clip_length
            final_clips.append({
                "title": f"Chat explosion at {spike['timestamp_display']}",
                "start_time": seconds_to_display(spike_ts),
                "end_time": seconds_to_display(end_ts),
                "start_seconds": spike_ts,
                "end_seconds": end_ts,
                "duration_seconds": clip_length,
                "total_score": round(chat_only_score, 1),
                "confidence_tier": "⚡ High" if chat_only_score >= 60 else "✅ Medium",
                "gemini_score": 0,
                "chat_confidence": spike["confidence"],
                "has_community_timestamp": False,
                "has_superchat": "SUPER CHAT" in " ".join(spike.get("sample_messages", [])).upper(),
                "suggested_caption": "Chat went crazy here 👀",
                "hook_line": "Nobody expected this...",
                "clip_type": "viral",
                "energy_level": "high",
                "reason": f"Massive chat spike — {spike['message_count']} messages in 10 seconds",
                "signals": [f"Chat spike x{spike['spike_ratio']} ({spike['message_count']} msgs)"],
                "both_signals": False,
            })

    # Sort by total score and limit to max_clips
    final_clips.sort(key=lambda x: x["total_score"], reverse=True)
    final_clips = final_clips[:max_clips]

    # Add final rank
    for i, clip in enumerate(final_clips, 1):
        clip["rank"] = i

    return final_clips


def print_final_clips(clips: list, video_title: str = ""):
    """Pretty prints the final combined clip rankings."""
    print(f"\n{'═' * 60}")
    print(f"CLIPFORGE FINAL RESULTS")
    if video_title:
        print(f"Video: {video_title}")
    print(f"{'═' * 60}")

    for clip in clips:
        both = "← BOTH SIGNALS AGREE 🎯" if clip["both_signals"] else ""
        print(f"\n  #{clip['rank']} — {clip['title']}")
        print(f"  Time        : {clip['start_time']} → {clip['end_time']}")
        print(f"  Score       : {clip['total_score']}/115  {clip['confidence_tier']}")
        print(f"  Gemini      : {clip['gemini_score']}/100")
        print(f"  Chat        : {clip['chat_confidence']}% confidence")
        print(f"  Community ts: {'Yes ✓' if clip['has_community_timestamp'] else 'No'}")
        print(f"  {both}")
        print(f"  Caption     : \"{clip['suggested_caption']}\"")
        print(f"  Hook        : \"{clip['hook_line']}\"")
        print(f"  Why         : {clip['reason']}")
        print(f"  Signals     :")
        for signal in clip["signals"]:
            print(f"    • {signal}")

    print(f"\n{'─' * 60}")
    dual_count = sum(1 for c in clips if c["both_signals"])
    print(f"Clips with dual signal confirmation: {dual_count}/{len(clips)}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    # Demo mode: load from saved JSON files
    import sys

    gemini_file = sys.argv[1] if len(sys.argv) > 1 else "last_results.json"
    chat_file = sys.argv[2] if len(sys.argv) > 2 else "chat_spikes.json"

    try:
        with open(gemini_file) as f:
            gemini_results = json.load(f)
        with open(chat_file) as f:
            chat_results = json.load(f)

        clips = combine_signals(gemini_results, chat_results)
        print_final_clips(clips, gemini_results.get("source_title", ""))

        with open("final_clips.json", "w") as f:
            json.dump(clips, f, indent=2, ensure_ascii=False)
        print("Saved to: final_clips.json")

    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("Run pipeline.py and chat_spike_detector.py first to generate the input files.")
