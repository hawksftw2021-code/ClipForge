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
    """Serve the ClipForge frontend — no cache."""
    html_path = Path("clipforge.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        from flask import Response
        response = Response(html, mimetype="text/html")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
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

    # Block obviously invalid URLs
    # yt-dlp supports YouTube, Twitch, Kick, Twitter, Facebook, Vimeo and 1000+ more
    BLOCKED = ["netflix.com", "spotify.com", "apple.com/music", "tidal.com"]
    if any(b in url for b in BLOCKED):
        return jsonify({"error": "This platform is not supported. Try YouTube, Twitch, Kick, or Vimeo."}), 400
    if not url.startswith("http"):
        return jsonify({"error": "Please provide a valid URL starting with http or https"}), 400

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


# ─── FILE UPLOAD ENDPOINT ────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a direct video file upload from the creator.
    Saves it temporarily and runs through the pipeline.
    No yt-dlp needed — creator uploads their own content.

    Form data:
      file       — video file (MP4, MOV, AVI, WebM)
      clip_type  — viral | highlights | hooks | funny | tips | quotable
      num_clips  — 1-10
      clip_length — 15-180
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file       = request.files["file"]
    clip_type  = request.form.get("clip_type",   "viral")
    num_clips  = int(request.form.get("num_clips",  5))
    clip_length = int(request.form.get("clip_length", 45))

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    allowed = {"mp4", "mov", "avi", "webm", "mkv", "m4a", "m4v"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"File type .{ext} not supported. Use MP4, MOV, AVI, or WebM"}), 400

    # MIME type mapping
    mime_map = {
        "mp4":  "video/mp4",
        "mov":  "video/quicktime",
        "avi":  "video/avi",
        "webm": "video/webm",
        "mkv":  "video/webm",
        "m4v":  "video/mp4",
        "m4a":  "audio/mp4",
    }
    mime_type = mime_map.get(ext, "video/mp4")

    import tempfile
    import os

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path  = tmp.name
            file_size = os.path.getsize(tmp_path) / 1024 / 1024

        print(f"[ClipForge] Direct upload: {file.filename} ({file_size:.1f}MB)")
        print(f"[ClipForge] Type: {clip_type} | Clips: {num_clips} | Length: {clip_length}s")

        # Run pipeline directly on the uploaded file
        from pipeline import analyze_with_gemini, run_trend_and_scoring
        results = run_trend_and_scoring(
            audio_path=tmp_path,
            mime_type=mime_type,
            clip_type=clip_type,
            num_clips=num_clips,
            clip_length=clip_length,
            source_url=f"upload://{file.filename}",
            title=file.filename.rsplit(".", 1)[0],
        )

        # Clean up temp file
        os.unlink(tmp_path)

        # Load cost info
        cost_info = {}
        if os.path.exists("cost_log.json"):
            with open("cost_log.json", "r") as f:
                log = json.load(f)
                cost_info = {
                    "this_run":    log["runs"][-1]["cost_usd"] if log["runs"] else 0,
                    "total_spent": log["total_spent_usd"],
                    "total_runs":  log["total_runs"],
                }

        return jsonify({
            "success":          True,
            "title":            results.get("source_title", file.filename),
            "language":         results.get("video_language", ""),
            "summary":          results.get("video_summary", ""),
            "clips":            results.get("clips", []),
            "best_clip_reason": results.get("best_clip_reason", ""),
            "cost":             cost_info,
            "source_url":       f"upload://{file.filename}",
        })

    except Exception as e:
        print(f"[ClipForge] Upload error: {e}")
        # Clean up if error
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
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
    burn_captions = data.get("burn_captions", False)

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

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None  # debug off in production
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )
