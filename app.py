"""
ClipForge — Flask Backend
--------------------------
The server that connects the UI to the pipeline.

Routes:
  GET  /                    → serves the frontend UI
  POST /analyze             → runs Gemini analysis, returns clip suggestions
  POST /clip                → runs FFmpeg clipper, returns downloadable clips
  GET  /clips/<filename>    → serves a generated clip file
  GET  /status              → health check

Usage:
  python app.py
  Then open http://localhost:5000 in your browser
"""

import os
import json
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

# Import our pipeline modules
from pipeline import run_pipeline
from clipper import run_clipper

load_dotenv()

app = Flask(__name__)
CORS(app)

# Where generated clips are stored
CLIPS_DIR = Path("clips")
CLIPS_DIR.mkdir(exist_ok=True)

# In-memory job tracker — stores progress and results per job
# In production this would be Redis or a database
jobs = {}


# ─── SERVE FRONTEND ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the ClipForge frontend."""
    html_path = Path("clipforge.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "ClipForge — clipforge.html not found", 404


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "version": "1.0.0",
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "youtube_key": bool(os.getenv("YOUTUBE_API_KEY")),
    })


# ─── ANALYZE ENDPOINT ─────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts a YouTube URL and settings.
    Runs the Gemini pipeline and returns clip suggestions.

    Request body (JSON):
    {
        "url": "https://youtube.com/watch?v=...",
        "clip_type": "viral",
        "num_clips": 5,
        "clip_length": 45
    }
    """
    data = request.get_json()

    # Validate input
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Please provide a valid YouTube URL"}), 400

    clip_type  = data.get("clip_type", "viral")
    num_clips  = int(data.get("num_clips", 5))
    clip_length = int(data.get("clip_length", 45))

    # Validate clip type
    valid_types = ["viral", "highlights", "hooks", "funny", "tips", "quotable"]
    if clip_type not in valid_types:
        clip_type = "viral"

    # Clamp values to safe ranges
    num_clips   = max(1, min(10, num_clips))
    clip_length = max(15, min(90, clip_length))

    try:
        print(f"\n[ClipForge] Analyzing: {url}")
        print(f"[ClipForge] Type: {clip_type} | Clips: {num_clips} | Length: {clip_length}s")

        # Run the pipeline
        results = run_pipeline(url, clip_type, num_clips, clip_length)

        # Load cost log for this response
        cost_info = {}
        if os.path.exists("cost_log.json"):
            with open("cost_log.json", "r") as f:
                log = json.load(f)
                cost_info = {
                    "this_run": log["runs"][-1]["cost_usd"] if log["runs"] else 0,
                    "total_spent": log["total_spent_usd"],
                    "total_runs": log["total_runs"],
                }

        return jsonify({
            "success": True,
            "title": results.get("source_title", ""),
            "language": results.get("video_language", ""),
            "summary": results.get("video_summary", ""),
            "clips": results.get("clips", []),
            "best_clip_reason": results.get("best_clip_reason", ""),
            "cost": cost_info,
            "source_url": url,
        })

    except Exception as e:
        print(f"[ClipForge] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── CLIP ENDPOINT ────────────────────────────────────────────────────────────

@app.route("/clip", methods=["POST"])
def clip():
    """
    Takes analysis results and cuts the actual video clips.
    Runs FFmpeg and returns download links.

    Request body (JSON):
    {
        "source_url": "https://youtube.com/watch?v=...",
        "clips": [...],  // from /analyze response
        "format": "vertical",
        "burn_captions": true
    }
    """
    data = request.get_json()

    source_url    = data.get("source_url", "")
    clips         = data.get("clips", [])
    fmt           = data.get("format", "vertical")
    burn_captions = data.get("burn_captions", True)

    if not source_url or not clips:
        return jsonify({"error": "Missing source URL or clips"}), 400

    valid_formats = ["vertical", "horizontal", "square"]
    if fmt not in valid_formats:
        fmt = "vertical"

    try:
        # Build a temporary results dict for the clipper
        results = {
            "source_url":   source_url,
            "source_title": data.get("title", "video"),
            "clips":        clips,
        }

        # Save temp results file for clipper
        temp_file = f"temp_results_{uuid.uuid4().hex[:8]}.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        # Run the clipper
        created_clips = run_clipper(temp_file, fmt, burn_captions)

        # Clean up temp file
        os.remove(temp_file)

        # Build download URLs for each clip
        clip_downloads = []
        for clip in created_clips:
            file_path = Path(clip["file"])
            if file_path.exists():
                clip_downloads.append({
                    "rank":        clip["rank"],
                    "title":       clip["title"],
                    "start_time":  clip["start_time"],
                    "end_time":    clip["end_time"],
                    "viral_score": clip["viral_score"],
                    "size_mb":     clip["size_mb"],
                    "hook_line":   clip.get("hook_line", ""),
                    "caption":     clip.get("caption", ""),
                    "download_url": f"/download/{file_path.as_posix()}",
                    "filename":    file_path.name,
                })

        return jsonify({
            "success": True,
            "clips_created": len(clip_downloads),
            "format": fmt,
            "clips": clip_downloads,
        })

    except Exception as e:
        print(f"[ClipForge] Clip error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── DOWNLOAD ENDPOINT ────────────────────────────────────────────────────────

@app.route("/download/<path:filepath>")
def download(filepath):
    """Serve a generated clip file for download."""
    file_path = Path(filepath)

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Security check — only serve files from clips directory
    try:
        file_path.resolve().relative_to(Path("clips").resolve())
    except ValueError:
        return jsonify({"error": "Access denied"}), 403

    return send_file(
        file_path,
        as_attachment=True,
        download_name=file_path.name,
        mimetype="video/mp4",
    )


# ─── COST ENDPOINT ────────────────────────────────────────────────────────────

@app.route("/costs")
def costs():
    """Returns current cost tracking data."""
    if not os.path.exists("cost_log.json"):
        return jsonify({"total_runs": 0, "total_spent_usd": 0, "runs": []})

    with open("cost_log.json", "r") as f:
        return jsonify(json.load(f))


# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  CLIPFORGE SERVER")
    print("═" * 50)
    print(f"  URL     : http://localhost:5000")
    print(f"  Gemini  : {'✓ Ready' if os.getenv('GEMINI_API_KEY') else '✗ Missing GEMINI_API_KEY'}")
    print(f"  YouTube : {'✓ Ready' if os.getenv('YOUTUBE_API_KEY') else '✗ Missing YOUTUBE_API_KEY'}")
    print("═" * 50 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
