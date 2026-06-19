"""
ClipForge — Scoring Engine
---------------------------
Multi-signal scoring system with citations and confidence weighting.

Signals:
  1. Gemini audio analysis (always available)
  2. Google NLP sentiment on raw transcript (paid, high quality)
  3. YouTube comment timestamp extraction (free, real audience data)
  4. Google Trends + YouTube trending (free, real trend data)
  5. Chat spike detector (free, livestream VODs only)

Weights:
  Trend     30%
  Hook      25%
  Audience  20%
  Energy    15%
  Value     10%

Confidence modifier:
  5 signals → score x 1.00
  4 signals → score x 0.95
  3 signals → score x 0.90
  2 signals → score x 0.82
  1 signal  → score x 0.75
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_NLP_API_KEY = os.getenv("GOOGLE_NLP_API_KEY")
YOUTUBE_API_KEY    = os.getenv("YOUTUBE_API_KEY")

# ─── BUZZWORD LIBRARY ────────────────────────────────────────────────────────
# Proven engagement triggers by language and content type
# These words in transcript = audience reaction signal
# Weighted by reaction intensity

BUZZWORDS = {
    "BR": {
        "viral": [
            # Reaction spikes
            "kkkkkk", "kkkkkkk", "kkkkkkkk", "kkkk", "kkkkk",
            "mds", "meu deus", "gente", "oxi", "oxente",
            "que isso", "que absurdo", "não acredito", "impossível",
            "caralho", "caramba", "nossa senhora", "nossa",
            "clipa", "clipa isso", "clipa aí", "clipa esse",
            "que jogada", "inacreditável", "absurdo", "loucura",
            "tô morto", "morri", "morre", "socorro",
            "verdade", "fatos", "real", "exatamente",
        ],
        "funny": [
            "kkkkk", "kkkkkk", "hahaha", "ahahaha",
            "tô morto", "morri de rir", "socorro", "morre",
            "que piada", "hilário", "engraçado", "ridículo",
            "meu deus", "gente", "oxi", "que isso",
        ],
        "gaming": [
            "clip isso", "clipou", "que jogada", "impossível",
            "gg", "ggwp", "fodou", "que aim", "que skill",
            "kkkkkk", "mds", "que isso", "inacreditável",
            "não acredito", "caralho", "nossa",
        ],
        "podcast": [
            "verdade", "fatos", "concordo", "discordo",
            "polêmico", "interessante", "nunca pensei",
            "meu deus", "gente", "que isso", "absurdo",
            "real", "exatamente", "com certeza",
        ],
        "hooks": [
            "você não vai acreditar", "ninguém esperava",
            "isso mudou tudo", "a verdade sobre",
            "nunca te contaram", "o segredo",
            "gente olha isso", "meu deus olha",
        ],
    },
    "BR_sports": {
        "viral": [
            # Soccer reaction words
            "goool", "gol", "que gol", "que golaço", "golaço",
            "que jogada", "incrível", "impossível", "que isso",
            "meu deus", "gente", "oxi", "kkkkkk", "mds",
            "não acredito", "absurdo", "inacreditável",
            "que dribles", "que passe", "assistência",
            "passou fácil", "humilhou", "dançou", "deixou no chão",
            "que finalização", "no ângulo", "explodiu o gol",
            "defesa incrível", "que defesa", "voou",
            "pênalti", "gol de falta", "golaço de falta",
            "vamo brasil", "brasil", "seleção",
            "flamengo", "corinthians", "palmeiras", "são paulo",
            "neymar", "vini", "rodrygo", "endrick",
        ],
        "funny": [
            "kkkkkk", "mds", "que isso", "gente",
            "caiu feio", "fingiu demais", "teatro",
            "simulou", "que queda", "dramático",
            "tá de brincadeira", "não pode isso",
        ],
        "hooks": [
            "você não vai acreditar", "nunca visto antes",
            "que momento", "histórico", "lendário",
            "melhor gol do ano", "impossível de defender",
            "ninguém esperava", "que surpresa",
        ],
    },
    "US_sports": {
        "viral": [
            "what a goal", "incredible", "no way", "insane",
            "unbelievable", "lets go", "goat move", "clip it",
            "did you see that", "filthy", "what a save",
            "top bins", "banger", "absolute rocket",
            "skill issue", "cooked him", "ankles broken",
            "nutmeg", "hat trick", "what a strike",
        ],
        "funny": [
            "he fell", "simulation", "oscar worthy",
            "he dove", "acting", "theatrical",
            "no way that hurt", "dramatic",
        ],
        "hooks": [
            "you won't believe", "never seen before",
            "goal of the year", "impossible to stop",
            "nobody expected", "historic moment",
        ],
    },
    "US": {
        "viral": [
            "clip it", "clip that", "no way", "bro what",
            "insane", "lets go", "lets goooo", "omg",
            "wait what", "hold on", "i cant", "dead",
            "im crying", "💀", "lmaooo", "lmfao",
            "nobody saw that", "did you see that",
            "this is crazy", "actual goat",
        ],
        "funny": [
            "dead", "im dead", "im crying", "i cant breathe",
            "lmaooo", "lmfao", "bro 💀", "no way bro",
            "this is hilarious", "i cant", "help",
        ],
        "gaming": [
            "clip it", "poggers", "pog", "gg", "ez",
            "sheesh", "insane", "lets go", "no way",
            "actual cheater", "that aim", "bro what",
        ],
        "podcast": [
            "facts", "real talk", "thats crazy",
            "i never thought", "wait really", "no way",
            "actually true", "based", "this", "literally",
        ],
        "hooks": [
            "you won't believe", "nobody expected",
            "this changed everything", "the truth about",
            "they never told you", "the secret",
            "wait for it", "watch what happens",
        ],
    },
}

# Buzzword weights — how much each hit boosts the score
BUZZWORD_WEIGHTS = {
    "viral":   3,   # strongest signal
    "funny":   2,
    "gaming":  2,
    "podcast": 1,
    "hooks":   3,
}


def detect_buzzwords(text: str, region: str, clip_type: str) -> dict:
    """
    Scans transcript text for buzzwords and returns hits with boost score.
    """
    if not text:
        return {"hits": [], "boost": 0, "citation": None}

    text_lower = text.lower()
    # Sports gets its own regional buzzword set
    if clip_type == "sports":
        bw_region = f"{region}_sports" if f"{region}_sports" in BUZZWORDS else "US_sports"
    else:
        bw_region = region
    region_words = BUZZWORDS.get(bw_region, BUZZWORDS.get("US", {}))

    hits     = []
    boost    = 0
    types_to_check = [clip_type, "viral", "hooks"]  # always check viral + hooks

    for bw_type in types_to_check:
        words  = region_words.get(bw_type, [])
        weight = BUZZWORD_WEIGHTS.get(bw_type, 1)
        for word in words:
            if word.lower() in text_lower:
                hits.append({"word": word, "type": bw_type, "weight": weight})
                boost += weight * 3  # 3 points per weight unit

    # Deduplicate hits
    seen  = set()
    unique_hits = []
    for h in hits:
        if h["word"] not in seen:
            seen.add(h["word"])
            unique_hits.append(h)

    boost = min(25, boost)  # cap at 25 point boost

    citation = None
    if unique_hits:
        top_words = [h["word"] for h in unique_hits[:3]]
        citation  = f"Buzzword detection: {', '.join(top_words)} — known viral triggers for {region} {clip_type} content"

    return {"hits": unique_hits, "boost": boost, "citation": citation}


WEIGHTS = {
    "trend":    0.30,
    "hook":     0.25,
    "audience": 0.20,
    "energy":   0.15,
    "value":    0.10,
}

CONFIDENCE_MODIFIERS = {5: 1.00, 4: 0.95, 3: 0.90, 2: 0.82, 1: 0.75}
MIN_COMMENTS = 50


def _ts_to_secs(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def _clip_transcript(transcript, start_secs, end_secs):
    if not transcript or not isinstance(transcript, list):
        return ""
    parts = []
    for seg in transcript:
        secs = _ts_to_secs(seg.get("timestamp", "0:00"))
        if start_secs - 10 <= secs <= end_secs + 10:
            parts.append(seg.get("text", ""))
    return " ".join(parts)


# ── GOOGLE NLP ────────────────────────────────────────────────────────────────

def analyze_nlp(text: str) -> dict:
    if not GOOGLE_NLP_API_KEY or not text:
        return {"available": False, "reason": "No NLP key"}
    url = f"https://language.googleapis.com/v1/documents:analyzeSentiment?key={GOOGLE_NLP_API_KEY}"
    try:
        r = requests.post(url, json={
            "document": {"type": "PLAIN_TEXT", "content": text[:10000]},
            "encodingType": "UTF8"
        }, timeout=10)
        data = r.json()
        if "error" in data:
            return {"available": False, "reason": data["error"].get("message")}
        doc = data.get("documentSentiment", {})
        sentences = data.get("sentences", [])
        intense = sorted(sentences,
            key=lambda s: abs(s.get("sentiment",{}).get("score",0)) * s.get("sentiment",{}).get("magnitude",0),
            reverse=True)[:3]
        return {
            "available":  True,
            "score":      doc.get("score", 0),
            "magnitude":  doc.get("magnitude", 0),
            "sent_count": len(sentences),
            "intense":    [{"text": s["text"]["content"][:80],
                            "score": s["sentiment"]["score"],
                            "magnitude": s["sentiment"]["magnitude"]} for s in intense],
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def nlp_hook_score(nlp, clip_text):
    if not nlp.get("available"):
        return None, None
    mag = nlp.get("magnitude", 0)
    has_hook = any(c in clip_text[:200] for c in ["?", "!", "but", "actually", "wait"])
    score = min(100, int(50 + mag * 10 + (10 if has_hook else 0)))
    intense = nlp.get("intense", [])
    citation = f"Google NLP: emotional magnitude {mag:.1f}/10"
    if intense:
        citation += f'. Most intense: "{intense[0]["text"][:50]}..."'
    return score, citation


def nlp_energy_score(nlp):
    if not nlp.get("available"):
        return None, None
    mag = nlp.get("magnitude", 0)
    score = min(100, int(40 + mag * 8))
    if len(nlp.get("intense", [])) >= 3:
        score = min(100, score + 10)
    peaks = [s["magnitude"] for s in nlp.get("intense", [])]
    citation = f"Google NLP: {nlp.get('sent_count',0)} sentences analyzed, peak intensity {max(peaks):.1f}" if peaks else f"Google NLP: magnitude {mag:.1f}"
    return score, citation


# ── YOUTUBE COMMENTS ──────────────────────────────────────────────────────────

def get_comment_signals(video_id: str) -> dict:
    if not YOUTUBE_API_KEY:
        return {"available": False, "reason": "No YouTube key"}
    try:
        from googleapiclient.discovery import build
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        stats = yt.videos().list(part="statistics", id=video_id).execute()
        items = stats.get("items", [])
        if not items:
            return {"available": False, "reason": "Video not found"}
        count = int(items[0].get("statistics", {}).get("commentCount", 0))
        if count < MIN_COMMENTS:
            return {"available": False, "reason": f"Only {count} comments (min {MIN_COMMENTS})", "count": count}
        resp = yt.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, order="relevance"
        ).execute()
        pattern = re.compile(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b')
        ts_mentions = []
        top_comment = {"text": "", "likes": 0}
        for item in resp.get("items", []):
            snip  = item["snippet"]["topLevelComment"]["snippet"]
            text  = snip.get("textDisplay", "")
            likes = snip.get("likeCount", 0)
            if likes > top_comment["likes"]:
                top_comment = {"text": text[:100], "likes": likes}
            for m in pattern.finditer(text):
                parts = [g for g in m.groups() if g is not None]
                secs  = int(parts[0])*60+int(parts[1]) if len(parts)==2 else int(parts[0])*3600+int(parts[1])*60+int(parts[2])
                ctx   = text[max(0,m.start()-10):m.end()+50].strip()
                ts_mentions.append({"seconds": secs, "display": m.group(0), "context": ctx[:80], "likes": likes})
        ts_mentions.sort(key=lambda x: x["likes"], reverse=True)
        return {"available": True, "count": count, "timestamps": ts_mentions[:10], "top_comment": top_comment}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def comment_audience_score(comment_data, start_secs, end_secs):
    if not comment_data.get("available"):
        return None, None
    mentions = [t for t in comment_data.get("timestamps", [])
                if start_secs - 15 <= t["seconds"] <= end_secs + 15]
    if not mentions:
        return 60, f"No timestamp mentions in {comment_data.get('count',0):,} comments for this window"
    total_likes = sum(m["likes"] for m in mentions)
    score = min(100, 60 + len(mentions)*8 + min(20, total_likes//10))
    top = max(mentions, key=lambda x: x["likes"])
    citation = f"{len(mentions)} viewer(s) referenced this moment in comments"
    if top["likes"] > 0:
        citation += f'. Top ({top["likes"]} likes): "{top["context"]}"'
    citation += " — Source: YouTube Data API"
    return score, citation


# ── MAIN SCORING ──────────────────────────────────────────────────────────────

def score_clip(clip, transcript, video_id, trend_data,
               comment_data=None, chat_spikes=None):
    start = _ts_to_secs(clip.get("start_time", "0:00"))
    end   = _ts_to_secs(clip.get("end_time",   "0:45"))
    clip_text = _clip_transcript(transcript, start, end)

    g = clip.get("scores", {})
    scores    = {k: g.get(k, 70) for k in WEIGHTS}
    citations = {
        "hook":     {"score": scores["hook"],     "source": "Gemini 3.5 Flash", "citation": clip.get("hook_explanation",   "AI hook quality assessment")},
        "energy":   {"score": scores["energy"],   "source": "Gemini 3.5 Flash", "citation": clip.get("energy_explanation", "AI energy level assessment")},
        "audience": {"score": scores["audience"], "source": "Gemini 3.5 Flash", "citation": "AI estimated audience reaction"},
        "value":    {"score": scores["value"],    "source": "Gemini 3.5 Flash", "citation": clip.get("value_explanation",  "AI value assessment")},
        "trend":    {"score": scores["trend"],    "source": "Gemini 3.5 Flash", "citation": "AI trend estimate"},
    }
    signals = ["Gemini"]

    # ── Buzzword detection ─────────────────────────────────────────────────
    region     = trend_data.get("region", "US") if trend_data else "US"
    bw_results = detect_buzzwords(clip_text, region, clip.get("clip_type", "viral"))
    if bw_results["boost"] > 0:
        scores["hook"]   = min(100, scores["hook"]   + bw_results["boost"] // 2)
        scores["energy"] = min(100, scores["energy"] + bw_results["boost"] // 2)
        if bw_results["citation"]:
            citations["hook"]["citation"] += f". {bw_results['citation']}"
        signals.append("Buzzwords")

    # ── Google NLP ─────────────────────────────────────────────────────────
    if clip_text and GOOGLE_NLP_API_KEY:
        nlp = analyze_nlp(clip_text)
        if nlp.get("available"):
            h_score, h_cite = nlp_hook_score(nlp, clip_text)
            e_score, e_cite = nlp_energy_score(nlp)
            if h_score:
                scores["hook"] = round(scores["hook"]*0.6 + h_score*0.4)
                citations["hook"] = {"score": scores["hook"], "source": "Gemini + Google NLP", "citation": h_cite}
            if e_score:
                scores["energy"] = round(scores["energy"]*0.6 + e_score*0.4)
                citations["energy"] = {"score": scores["energy"], "source": "Gemini + Google NLP", "citation": e_cite}
            signals.append("Google NLP")

    # ── YouTube Comments ───────────────────────────────────────────────────
    if comment_data and comment_data.get("available"):
        a_score, a_cite = comment_audience_score(comment_data, start, end)
        if a_score:
            scores["audience"] = round(scores["audience"]*0.5 + a_score*0.5)
            citations["audience"] = {"score": scores["audience"], "source": "Gemini + YouTube Comments", "citation": a_cite}
            signals.append("YouTube Comments")

    # ── Chat Spikes ────────────────────────────────────────────────────────
    if chat_spikes:
        nearby = [s for s in chat_spikes if abs(s.get("timestamp_seconds",0) - start) <= 30]
        if nearby:
            best  = max(nearby, key=lambda x: x.get("confidence", 0))
            spike = min(100, 60 + best.get("confidence",0)//2)
            scores["audience"] = round(scores["audience"]*0.5 + spike*0.5)
            prev = citations["audience"].get("citation","")
            citations["audience"] = {
                "score":   scores["audience"],
                "source":  "Gemini + Comments + Chat Spikes",
                "citation": prev + f'. Chat spiked {best.get("spike_ratio",1)}x at this timestamp ({best.get("message_count",0)} msgs/10s)',
            }
            signals.append("Chat Spikes")

    # ── Trend Data ─────────────────────────────────────────────────────────
    if trend_data and trend_data.get("trending_tags"):
        topics   = (clip.get("title","") + " " + clip.get("reason","")).lower().split()
        t_tags   = [t.lower() for t in trend_data.get("trending_tags",[])]
        t_search = [t.lower() for t in trend_data.get("trending_searches",[])]
        tag_hits    = [tg for topic in topics for tg in t_tags   if topic in tg or tg in topic]
        search_hits = [sr for topic in topics for sr in t_search if topic in sr or sr in topic]
        all_hits    = list(set(tag_hits + search_hits))
        if all_hits:
            boost = min(30, len(all_hits)*8)
            scores["trend"] = min(100, scores["trend"] + boost)
            cite = f"Matches {len(all_hits)} trending topic(s)"
            if tag_hits:    cite += f". YouTube trending: {', '.join(list(set(tag_hits))[:3])}"
            if search_hits: cite += f". Google Trends: {', '.join(list(set(search_hits))[:3])}"
            cite += f". Source: Google Trends + YouTube Data API ({trend_data.get('region','US')})"
            citations["trend"] = {"score": scores["trend"], "source": "Google Trends + YouTube", "citation": cite}
            signals.append("Trend Data")
        else:
            citations["trend"]["citation"] += " — no overlap with current trending content"

    # ── Confidence + Final Score ───────────────────────────────────────────
    n   = len(signals)
    mod = CONFIDENCE_MODIFIERS.get(n, 0.75)
    raw = sum(scores.get(k,50)*w for k,w in WEIGHTS.items())
    final = round(raw * mod)

    confidence_labels = {
        5: ("Very High", "All 5 signals confirmed this clip"),
        4: ("High",      "4 of 5 signals confirmed this clip"),
        3: ("Medium",    "3 signals — connect YouTube account for higher accuracy"),
        2: ("Low-Medium","2 signals — limited real-world data"),
        1: ("Low",       "AI analysis only — no real audience data available"),
    }
    conf_level, conf_note = confidence_labels.get(n, ("Low","AI only"))

    conflicts = []
    vals = list(scores.values())
    if max(vals) - min(vals) > 30:
        conflicts.append("Signals show mixed results — treat score with caution")

    return {
        **clip,
        "viral_score":        final,
        "scores":             scores,
        "citations":          citations,
        "signals_used":       signals,
        "signal_count":       n,
        "confidence_level":   conf_level,
        "confidence_note":    conf_note,
        "confidence_mod":     mod,
        "conflicts":          conflicts,
        "weighted_breakdown": {k: round(scores.get(k,50)*v,1) for k,v in WEIGHTS.items()},
        "hashtags":           clip.get("hashtags", []),
    }


def score_all_clips(clips, transcript, video_id, trend_data,
                    comment_data=None, chat_spikes=None):
    scored = [
        score_clip(c, transcript, video_id, trend_data, comment_data, chat_spikes)
        for c in clips
    ]
    scored.sort(key=lambda x: x["viral_score"], reverse=True)
    for i, c in enumerate(scored, 1):
        c["rank"] = i
    return scored
