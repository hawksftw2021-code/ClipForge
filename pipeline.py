"""
ClipForge Pipeline
------------------
Step 1: Download audio from YouTube URL via yt-dlp (no FFmpeg needed)
Step 2: Send audio to Gemini Flash for transcription + viral moment analysis
Step 3: Return structured clip suggestions with timestamps

Usage:
    python pipeline.py <youtube_url>
    python pipeline.py "https://www.youtube.com/watch?v=..."
"""

import os
import sys
import json
import glob
import tempfile
import datetime
import yt_dlp
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini 2.0 Flash pricing (per million tokens)
COST_PER_MILLION_INPUT  = 0.10
COST_PER_MILLION_OUTPUT = 0.40

COST_LOG_FILE = "cost_log.json"

# Gemini supports these audio formats natively — no FFmpeg needed
GEMINI_AUDIO_MIME_TYPES = {
    "webm": "audio/webm",
    "mp4":  "audio/mp4",
    "m4a":  "audio/mp4",
    "ogg":  "audio/ogg",
    "opus": "audio/ogg",
    "wav":  "audio/wav",
    "mp3":  "audio/mp3",
    "flac": "audio/flac",
}


# ─── COST TRACKING ────────────────────────────────────────────────────────────

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD from token counts."""
    input_cost  = (input_tokens  / 1_000_000) * COST_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT
    return round(input_cost + output_cost, 6)


def load_cost_log() -> dict:
    """Load existing cost log or create a fresh one."""
    if os.path.exists(COST_LOG_FILE):
        with open(COST_LOG_FILE, "r") as f:
            return json.load(f)
    return {
        "total_spent_usd": 0.0,
        "total_runs": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "runs": []
    }


def save_cost_entry(url: str, input_tokens: int, output_tokens: int, cost: float):
    """Append a cost entry to the log file."""
    log = load_cost_log()

    log["total_spent_usd"]      = round(log["total_spent_usd"] + cost, 6)
    log["total_runs"]           += 1
    log["total_input_tokens"]   += input_tokens
    log["total_output_tokens"]  += output_tokens

    log["runs"].append({
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url":            url,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "cost_usd":       cost,
        "running_total":  log["total_spent_usd"],
    })

    with open(COST_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def print_cost_summary(input_tokens: int, output_tokens: int, cost: float):
    """Print cost info after each run."""
    log = load_cost_log()
    print(f"\n{'─' * 60}")
    print(f"  COST TRACKER")
    print(f"{'─' * 60}")
    print(f"  This run:")
    print(f"    Input tokens  : {input_tokens:,}")
    print(f"    Output tokens : {output_tokens:,}")
    print(f"    Cost          : ${cost:.4f}")
    print(f"  All time:")
    print(f"    Total runs    : {log['total_runs']}")
    print(f"    Total spent   : ${log['total_spent_usd']:.4f}")
    print(f"{'─' * 60}\n")


# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

def download_audio(url: str, output_dir: str) -> tuple:
    """
    Downloads best available audio from YouTube URL.
    No FFmpeg required — downloads in native format, Gemini handles it.
    Returns (audio_path, title, duration, mime_type)
    """
    print(f"\n[1/3] Downloading audio from: {url}")

    output_path = os.path.join(output_dir, "audio.%(ext)s")

    ydl_opts = {
        "format":      "bestaudio/best",
        "outtmpl":     output_path,
        "quiet":       True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        title    = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        print(f"    Title   : {title}")
        print(f"    Duration: {duration // 60}m {duration % 60}s")

    files = glob.glob(os.path.join(output_dir, "audio.*"))
    if not files:
        raise FileNotFoundError("Audio download failed — no file found")

    audio_file = files[0]
    ext        = audio_file.rsplit(".", 1)[-1].lower()
    mime_type  = GEMINI_AUDIO_MIME_TYPES.get(ext, "audio/webm")

    print(f"    Format  : {ext} ({mime_type})")
    print(f"    Saved to: {audio_file}")
    return audio_file, title, duration, mime_type


# ─── GEMINI ANALYSIS ──────────────────────────────────────────────────────────

def analyze_with_gemini(
    audio_path: str,
    mime_type: str,
    clip_type: str = "viral",
    num_clips: int = 5,
    clip_length: int = 45,
    source_url: str = "",
) -> dict:
    """
    Sends audio to Gemini Flash for transcription + viral moment detection.
    Tracks token usage and cost automatically.
    Returns structured JSON with transcript and clip suggestions.
    """
    print(f"\n[2/3] Sending to Gemini Flash for analysis...")
    print(f"    Clip type  : {clip_type}")
    print(f"    Num clips  : {num_clips}")
    print(f"    Clip length: ~{clip_length}s each")

    client = genai.Client(api_key=GEMINI_API_KEY)

    print("    Uploading audio to Gemini...")
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    clip_type_instructions = {
        "viral":      "moments where energy spikes, crowd reacts loudly, speaker raises voice, laughter erupts, or something shocking/surprising happens",
        "highlights": "the most valuable, insightful, or impressive moments that best represent the content",
        "hooks":      "the most attention-grabbing opening moments that would make someone stop scrolling — prioritize the first 3 seconds of each potential clip",
        "funny":      "the funniest moments, unexpected reactions, jokes that landed, or comedic timing",
        "tips":       "practical tips, advice, how-to moments, product explanations, and any moment where the speaker shares genuinely useful information the viewer can act on",
        "quotable":   "single powerful statements, hot takes, memorable one-liners, strong opinions, or anything someone would screenshot or share as a quote — must stand alone without context",
    }

    clip_instruction = clip_type_instructions.get(clip_type, clip_type_instructions["viral"])

    prompt = f"""You are an expert video clip editor for social media. Analyze this audio and do two things:

1. TRANSCRIBE the full audio with accurate timestamps in [MM:SS] format
2. IDENTIFY the {num_clips} best clip moments for short-form content

For clip selection, focus on: {clip_instruction}

For each clip, look for:
- Energy spikes (volume, excitement, crowd noise)
- Emotional peaks (laughter, shock, hype, emotion)
- Standalone moments (make sense without full context)
- Hook potential (would stop someone scrolling)

Respond ONLY with valid JSON in this exact format, no other text:

{{
  "video_language": "detected language (e.g. English, Portuguese)",
  "transcript": [
    {{"timestamp": "0:00", "text": "transcribed text here"}},
    {{"timestamp": "0:15", "text": "more text here"}}
  ],
  "clips": [
    {{
      "rank": 1,
      "title": "Short punchy clip title",
      "start_time": "2:14",
      "end_time": "2:59",
      "duration_seconds": 45,
      "viral_score": 92,
      "clip_type": "viral",
      "reason": "One sentence explaining why this moment is great",
      "energy_level": "high",
      "suggested_caption": "Caption text to overlay on the clip",
      "hook_line": "First few words that grab attention"
    }}
  ],
  "video_summary": "2-3 sentence summary of the full video content",
  "best_clip_reason": "Why the #1 clip is the strongest"
}}

Target clip length: approximately {clip_length} seconds each.
Return exactly {num_clips} clips ranked by viral potential."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(data=audio_data, mime_type=mime_type),
            prompt,
        ]
    )

    # ── Token tracking ─────────────────────────────────────────────────────
    usage         = response.usage_metadata
    input_tokens  = getattr(usage, "prompt_token_count",     0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    cost          = calculate_cost(input_tokens, output_tokens)

    save_cost_entry(source_url, input_tokens, output_tokens, cost)
    print_cost_summary(input_tokens, output_tokens, cost)
    # ───────────────────────────────────────────────────────────────────────

    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    if raw.endswith("```"):
        raw = raw[:-3]

    result = json.loads(raw.strip())
    print(f"    Language detected  : {result.get('video_language', 'Unknown')}")
    print(f"    Transcript segments: {len(result.get('transcript', []))}")
    print(f"    Clips identified   : {len(result.get('clips', []))}")

    return result


# ─── RESULTS ──────────────────────────────────────────────────────────────────

def print_results(results: dict, title: str):
    """Pretty prints the clip suggestions to the terminal."""
    print(f"\n[3/3] Results for: {title}")
    print("=" * 60)
    print(f"\nVideo summary:\n  {results.get('video_summary', 'N/A')}")
    print(f"\nLanguage: {results.get('video_language', 'Unknown')}")
    print(f"\n{'─' * 60}")
    print(f"TOP CLIPS:")
    print(f"{'─' * 60}")

    for clip in results.get("clips", []):
        viral_bar = "█" * (clip["viral_score"] // 10) + "░" * (10 - clip["viral_score"] // 10)
        print(f"\n  #{clip['rank']} — {clip['title']}")
        print(f"  Time    : {clip['start_time']} → {clip['end_time']} ({clip['duration_seconds']}s)")
        print(f"  Score   : [{viral_bar}] {clip['viral_score']}/100")
        print(f"  Energy  : {clip['energy_level'].upper()}")
        print(f"  Why     : {clip['reason']}")
        print(f"  Caption : \"{clip['suggested_caption']}\"")
        print(f"  Hook    : \"{clip['hook_line']}\"")

    print(f"\n{'─' * 60}")
    print(f"Best clip: {results.get('best_clip_reason', 'N/A')}")
    print(f"{'─' * 60}\n")


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_pipeline(
    url: str,
    clip_type: str = "viral",
    num_clips: int = 5,
    clip_length: int = 45,
) -> dict:
    """Full ClipForge pipeline. Returns the complete results dict."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path, title, duration, mime_type = download_audio(url, tmp_dir)

        results = analyze_with_gemini(
            audio_path, mime_type, clip_type, num_clips, clip_length, source_url=url
        )
        results["source_title"]    = title
        results["source_duration"] = duration
        results["source_url"]      = url

        print_results(results, title)
        return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python pipeline.py <youtube_url> [clip_type] [num_clips] [clip_length]")
        print("  clip_type  : viral | highlights | hooks | funny  (default: viral)")
        print("  num_clips  : 1-10                                (default: 5)")
        print("  clip_length: seconds                             (default: 45)")
        print("\nExample:")
        print('  python pipeline.py "https://youtube.com/watch?v=xyz" viral 5 45\n')
        sys.exit(1)

    url         = sys.argv[1]
    clip_type   = sys.argv[2] if len(sys.argv) > 2 else "viral"
    num_clips   = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    clip_length = int(sys.argv[4]) if len(sys.argv) > 4 else 45

    results = run_pipeline(url, clip_type, num_clips, clip_length)

    with open("last_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: last_results.json\n")
