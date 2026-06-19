"""
ClipForge — Video Clipper
--------------------------
Takes the results from pipeline.py and uses FFmpeg to cut
actual video clips at the timestamps Gemini identified.

Produces MP4 files ready to post to TikTok, Reels, Shorts.

Format options:
  vertical   — 9:16 portrait with blurred background (default, TikTok/Reels/Shorts)
  horizontal — 16:9 landscape (YouTube, Twitter)
  square     — 1:1 (Instagram feed)

Usage:
    python clipper.py                          # vertical with blurred bg (default)
    python clipper.py last_results.json vertical
    python clipper.py last_results.json horizontal
    python clipper.py last_results.json square
    python clipper.py last_results.json vertical --no-captions
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path

OUTPUT_DIR = "clips"

# Output format definitions
FORMATS = {
    "vertical": {
        "width":       1080,
        "height":      1920,
        "description": "9:16 Vertical — TikTok, Reels, Shorts",
    },
    "horizontal": {
        "width":       1920,
        "height":      1080,
        "description": "16:9 Horizontal — YouTube, Twitter",
    },
    "square": {
        "width":       1080,
        "height":      1080,
        "description": "1:1 Square — Instagram feed",
    },
}


def download_video(url: str, output_dir: str) -> tuple:
    """
    Downloads the full video from YouTube.
    Returns (path, title)
    """
    import yt_dlp

    print(f"\n[1/3] Downloading full video...")
    print(f"      This may take a minute depending on video length.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source.%(ext)s")

    ydl_opts = {
        "format":      "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":     output_path,
        "quiet":       True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        title    = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        print(f"      Title   : {title}")
        print(f"      Duration: {duration // 60}m {duration % 60}s")

    files = glob.glob(os.path.join(output_dir, "source.*"))
    if not files:
        raise FileNotFoundError("Video download failed")

    video_file = files[0]
    size_mb    = os.path.getsize(video_file) / 1024 / 1024
    print(f"      File    : {video_file} ({size_mb:.1f} MB)")
    return video_file, title


def timestamp_to_seconds(ts: str) -> float:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


def build_ffmpeg_filter(fmt: str, caption: str = "") -> str:
    """
    Builds the FFmpeg video filter string for the chosen format.

    Vertical/Square: blurred background + centered video on top.
    This is the professional look used by most viral clip tools.

    Horizontal: just scale to fit with padding if needed.
    """
    w = FORMATS[fmt]["width"]
    h = FORMATS[fmt]["height"]

    if fmt == "vertical" or fmt == "square":
        # Fixed blurred background — works on webm and all source formats
        # Uses gblur instead of boxblur, split filter for dual stream
        bg_filter = (
            f"[0:v]format=yuv420p,split=2[bg_src][fg_src];"
            f"[bg_src]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},"
            f"gblur=sigma=20[bg];"
            f"[fg_src]scale={w}:{h}:force_original_aspect_ratio=decrease[fg_scaled];"
            f"[fg_scaled]pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black@0[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )

        if caption:
            safe = caption.replace("'", "\\'").replace(":", "\\:")[:80]
            filter_str = (
                f"{bg_filter};"
                f"[base]drawtext=text='{safe}'"
                f":fontsize=36"
                f":fontcolor=white"
                f":bordercolor=black"
                f":borderw=3"
                f":x=(w-text_w)/2"
                f":y=h-th-60[out]"
            )
            return filter_str, "[out]"
        return bg_filter.replace("[base]", "[out]"), "[out]"

    else:
        # Horizontal — scale to fit, pad with black if needed
        base_filter = (
            f"[0:v]format=yuv420p,"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[base]"
        )

        if caption:
            safe = caption.replace("'", "\\'").replace(":", "\\:")[:80]
            filter_str = (
                f"{base_filter};"
                f"[base]drawtext=text='{safe}'"
                f":fontsize=28"
                f":fontcolor=white"
                f":bordercolor=black"
                f":borderw=2"
                f":x=(w-text_w)/2"
                f":y=h-th-40[out]"
            )
            return filter_str, "[out]"
        return base_filter.replace("[base]", "[out]"), "[out]"


def cut_clip(
    source_video: str,
    start_time:   str,
    end_time:     str,
    output_path:  str,
    fmt:          str = "vertical",
    caption:      str = "",
) -> bool:
    """
    Uses FFmpeg to cut and reformat a clip.
    Returns True if successful.
    """
    start_secs    = timestamp_to_seconds(start_time)
    duration_secs = timestamp_to_seconds(end_time) - start_secs

    filter_str, map_label = build_ffmpeg_filter(fmt, caption)

    cmd = [
        "ffmpeg",
        "-ss",    str(start_secs),
        "-i",     source_video,
        "-t",     str(duration_secs),
        "-filter_complex", filter_str,
        "-map",   map_label,
        "-map",   "0:a",
        "-c:v",   "libx264",
        "-c:a",   "aac",
        "-preset","fast",
        "-crf",   "23",
        "-y",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        # Print FFmpeg error for debugging
        print(f"  FFmpeg error: {result.stderr.decode()[-300:]}")

    return result.returncode == 0


def cut_all_clips(
    source_video:  str,
    clips:         list,
    output_dir:    str,
    fmt:           str  = "vertical",
    burn_captions: bool = False,
) -> list:
    """
    Cuts all clips from the source video.
    Returns list of successfully created clip paths.
    """
    fmt_info = FORMATS[fmt]
    print(f"\n[2/3] Cutting {len(clips)} clips...")
    print(f"      Format : {fmt_info['description']}")
    print(f"      Size   : {fmt_info['width']}x{fmt_info['height']}")

    os.makedirs(output_dir, exist_ok=True)
    created = []

    for clip in clips:
        rank        = clip.get("rank", 0)
        title       = clip.get("title", f"clip_{rank}")
        start_time  = clip.get("start_time", "0:00")
        end_time    = clip.get("end_time", "0:45")
        caption     = clip.get("suggested_caption", "") if burn_captions else ""
        viral_score = clip.get("viral_score", 0)

        safe_title  = "".join(c for c in title if c.isalnum() or c in " -_")[:40].strip()
        filename    = f"clip_{rank:02d}_{fmt}_{safe_title}.mp4"
        output_path = os.path.join(output_dir, filename)

        print(f"\n  Clip #{rank}: {title}")
        print(f"  Time   : {start_time} → {end_time}")
        print(f"  Score  : {viral_score}/100")
        print(f"  Format : {fmt_info['description']}")

        success = cut_clip(source_video, start_time, end_time, output_path, fmt, caption)

        if success:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  ✓ Done : {filename} ({size_mb:.1f} MB)")
            created.append({
                "rank":        rank,
                "title":       title,
                "file":        output_path,
                "size_mb":     round(size_mb, 1),
                "format":      fmt,
                "start_time":  start_time,
                "end_time":    end_time,
                "viral_score": viral_score,
                "caption":     caption,
                "hook_line":   clip.get("hook_line", ""),
            })
        else:
            print(f"  ✗ Failed")

    return created


def print_summary(created_clips: list, output_dir: str, fmt: str):
    """Print final summary."""
    fmt_info = FORMATS[fmt]
    print(f"\n{'═' * 60}")
    print(f"CLIPFORGE — CLIPS READY")
    print(f"{'═' * 60}")
    print(f"Format  : {fmt_info['description']}")
    print(f"Location: {os.path.abspath(output_dir)}")
    print(f"Total   : {len(created_clips)} clips\n")

    for clip in created_clips:
        print(f"  #{clip['rank']} — {clip['title']}")
        print(f"       File  : {os.path.basename(clip['file'])}")
        print(f"       Time  : {clip['start_time']} → {clip['end_time']}")
        print(f"       Score : {clip['viral_score']}/100")
        print(f"       Size  : {clip['size_mb']} MB")
        print(f"       Hook  : \"{clip['hook_line']}\"")
        print()

    print(f"{'─' * 60}")
    print(f"Ready to post to TikTok, Reels, and Shorts!")
    print(f"{'─' * 60}\n")


def run_clipper(
    results_file:  str  = "last_results.json",
    fmt:           str  = "vertical",
    burn_captions: bool = True,
):
    """Full clipping pipeline."""
    if not os.path.exists(results_file):
        print(f"\nError: {results_file} not found.")
        print("Run pipeline.py first.\n")
        sys.exit(1)

    if fmt not in FORMATS:
        print(f"\nError: Unknown format '{fmt}'")
        print(f"Choose from: {', '.join(FORMATS.keys())}\n")
        sys.exit(1)

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    url   = results.get("source_url", "")
    clips = results.get("clips", [])
    title = results.get("source_title", "video")

    if not url or not clips:
        print("\nError: Invalid results file.")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"CLIPFORGE — VIDEO CLIPPER")
    print(f"{'═' * 60}")
    print(f"Video  : {title}")
    print(f"Clips  : {len(clips)}")
    print(f"Format : {FORMATS[fmt]['description']}")
    print(f"{'═' * 60}")

    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:30].strip()
    output_dir = os.path.join(OUTPUT_DIR, safe_title)

    source_video, _ = download_video(url, output_dir)
    created_clips   = cut_all_clips(source_video, clips, output_dir, fmt, burn_captions)

    print_summary(created_clips, output_dir, fmt)

    manifest_path = os.path.join(output_dir, f"clips_manifest_{fmt}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(created_clips, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved to: {manifest_path}\n")

    return created_clips


if __name__ == "__main__":
    results_file  = "last_results.json"
    fmt           = "vertical"
    burn_captions = True

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) >= 1 and args[0].endswith(".json"):
        results_file = args[0]
        if len(args) >= 2:
            fmt = args[1]
    elif len(args) >= 1:
        fmt = args[0]

    if "--no-captions" in sys.argv:
        burn_captions = False
        print("Caption burning disabled.")

    print(f"\nFormat selected: {FORMATS.get(fmt, {}).get('description', fmt)}")
    run_clipper(results_file, fmt, burn_captions)
