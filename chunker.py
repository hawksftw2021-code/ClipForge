"""
ClipForge — Audio Chunker
--------------------------
Splits long audio files into overlapping chunks for analysis.
Each chunk is analyzed independently by Gemini, then results
are merged and re-ranked into one final clip list.

Why overlapping chunks?
  A good moment might start near the end of one chunk and
  finish in the next. 60-second overlap prevents us from
  missing clips that cross chunk boundaries.

Usage:
  from chunker import chunk_audio, merge_chunk_results
  chunks = chunk_audio(audio_path, chunk_minutes=30, overlap_seconds=60)
  # analyze each chunk...
  final_clips = merge_chunk_results(all_chunk_results)
"""

import os
import subprocess
import tempfile
import math
from pathlib import Path


CHUNK_MINUTES   = 30    # analyze 30 minutes at a time
OVERLAP_SECONDS = 60    # 60 second overlap between chunks
MIN_CHUNK_SECS  = 300   # don't chunk videos under 5 minutes
MAX_SINGLE_SECS = 2700  # 45 minutes — above this we start chunking


def get_audio_duration(audio_path: str) -> float:
    """Returns audio duration in seconds using FFmpeg."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ], capture_output=True, text=True)

    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def chunk_audio(
    audio_path: str,
    chunk_minutes: int = CHUNK_MINUTES,
    overlap_seconds: int = OVERLAP_SECONDS,
) -> list:
    """
    Splits audio into overlapping chunks.

    Returns list of dicts:
    [
        {
            "path":        "/tmp/chunk_0.webm",
            "start_secs":  0,
            "end_secs":    1860,
            "chunk_index": 0,
            "total_chunks": 3,
            "offset_secs": 0,   # add to timestamps to get real video time
        },
        ...
    ]
    """
    duration = get_audio_duration(audio_path)

    if duration < MIN_CHUNK_SECS:
        # Very short — process as one chunk, no splitting needed
        print(f"    Short video ({duration/60:.1f}min) — processing as single chunk")
        return [{
            "path":         audio_path,
            "start_secs":   0,
            "end_secs":     duration,
            "chunk_index":  0,
            "total_chunks": 1,
            "offset_secs":  0,
            "needs_cleanup": False,
        }]

    if duration <= MAX_SINGLE_SECS:
        # Tweener (5-45 min) — Gemini handles this fine in one pass
        print(f"    Medium video ({duration/60:.1f}min) — processing as single chunk")
        return [{
            "path":         audio_path,
            "start_secs":   0,
            "end_secs":     duration,
            "chunk_index":  0,
            "total_chunks": 1,
            "offset_secs":  0,
            "needs_cleanup": False,
        }]

    chunk_secs  = chunk_minutes * 60
    ext         = Path(audio_path).suffix
    tmp_dir     = tempfile.mkdtemp(prefix="clipforge_chunks_")
    chunks      = []
    chunk_index = 0
    start       = 0

    print(f"\n    Chunking {duration/60:.1f} min audio into {chunk_minutes}-min chunks...")

    while start < duration:
        end    = min(start + chunk_secs + overlap_seconds, duration)
        out    = os.path.join(tmp_dir, f"chunk_{chunk_index}{ext}")

        # Use FFmpeg to cut this chunk
        subprocess.run([
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            out
        ], capture_output=True)

        chunks.append({
            "path":          out,
            "start_secs":    start,
            "end_secs":      end,
            "chunk_index":   chunk_index,
            "total_chunks":  0,  # filled in below
            "offset_secs":   start,
            "needs_cleanup": True,
            "tmp_dir":       tmp_dir,
        })

        print(f"    Chunk {chunk_index + 1}: {start/60:.1f}m \u2192 {end/60:.1f}m ({(end-start)/60:.1f}m)")

        # Advance — overlap means we go back overlap_seconds before end
        start      += chunk_secs
        chunk_index += 1

        if end >= duration:
            break

    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total

    print(f"    Split into {total} chunks")
    return chunks


def adjust_timestamps(clips: list, offset_secs: float) -> list:
    """
    Adjusts clip timestamps by adding the chunk offset.
    Converts "02:30" within a chunk to the real video timestamp.
    """
    adjusted = []
    for clip in clips:
        c = clip.copy()

        # Adjust start_time
        start = _ts_to_secs(c.get("start_time", "0:00")) + offset_secs
        end   = _ts_to_secs(c.get("end_time",   "0:45")) + offset_secs

        c["start_time"] = _secs_to_ts(start)
        c["end_time"]   = _secs_to_ts(end)
        c["chunk_offset"] = offset_secs

        adjusted.append(c)
    return adjusted


def merge_chunk_results(chunk_results: list, num_clips: int = 5) -> list:
    """
    Merges clip results from all chunks into one ranked list.
    Deduplicates clips that appear in overlapping sections.

    chunk_results: list of (result_dict, offset_secs) tuples
    """
    all_clips = []

    for result, offset in chunk_results:
        clips = result.get("clips", [])
        # Adjust timestamps to real video time
        adjusted = adjust_timestamps(clips, offset)
        all_clips.extend(adjusted)

    # Deduplicate — remove clips within 30 seconds of each other
    # (same moment detected in overlapping chunks)
    deduped = []
    for clip in sorted(all_clips, key=lambda x: x.get("viral_score", 0), reverse=True):
        clip_start = _ts_to_secs(clip.get("start_time", "0:00"))
        is_duplicate = False

        for existing in deduped:
            existing_start = _ts_to_secs(existing.get("start_time", "0:00"))
            if abs(clip_start - existing_start) < 30:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(clip)

        if len(deduped) >= num_clips * 2:
            break

    # Re-rank
    deduped.sort(key=lambda x: x.get("viral_score", 0), reverse=True)
    for i, clip in enumerate(deduped[:num_clips], 1):
        clip["rank"] = i

    return deduped[:num_clips]


def cleanup_chunks(chunks: list):
    """Removes temp chunk files after processing."""
    import shutil
    cleaned_dirs = set()
    for chunk in chunks:
        if chunk.get("needs_cleanup"):
            tmp_dir = chunk.get("tmp_dir")
            if tmp_dir and tmp_dir not in cleaned_dirs:
                try:
                    shutil.rmtree(tmp_dir)
                    cleaned_dirs.add(tmp_dir)
                    print(f"    Cleaned up temp chunks in {tmp_dir}")
                except Exception:
                    pass


def _ts_to_secs(ts: str) -> float:
    parts = str(ts).strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass
    return 0.0


def _secs_to_ts(secs: float) -> str:
    secs  = max(0, int(secs))
    h     = secs // 3600
    m     = (secs % 3600) // 60
    s     = secs % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
