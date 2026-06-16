"""Minimal local AI podcast generator: Ollama writes the script, Kokoro voices it,
the browser plays it. One file, no DB. See plan for design notes."""
import os
import warnings

# Quiet the third-party boot chatter (torch/diffusers/perth deprecations, HF telemetry)
# so startup prints just "Loading… / Ready". Doesn't affect our own logging.
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import json
import random
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, Response, request, send_file, send_from_directory

logging.getLogger().setLevel(logging.ERROR)  # mute root-logger model chatter; werkzeug keeps its own

app = Flask(__name__)
TMP = Path("/tmp/ai-podcast")
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

# Always two speakers: a woman on the left, a man on the right (matching the head
# models in static/heads/0.glb = female, 1.glb = male). Chatterbox clones a voice
# from a short reference clip, seeded once via Kokoro's named voices.
FEMALE = {"voices": ["af_heart", "bf_emma"], "names": ["Ava", "Sofia", "Mia", "Elena"]}
MALE = {"voices": ["am_michael", "bm_george"], "names": ["Marcus", "Leo", "Daniel", "Omar"]}
VOICES = FEMALE["voices"] + MALE["voices"]   # which refs to pre-seed at warmup
REF_DIR = TMP / "_refs"
REF_TEXT = ("Here's a quick thought on that — honestly, it depends, "
            "but I think it works out fine in the end.")
PERSONAS = [
    "a skeptical contrarian who pokes holes in every claim",
    "an optimistic futurist who sees the upside in everything",
    "a dry pragmatist focused on costs and trade-offs",
    "a curious newcomer who asks naive but sharp questions",
    "a data-driven analyst who wants evidence for everything",
    "a big-picture philosopher who zooms out to first principles",
    "a streetwise realist grounded in everyday experience",
    "an enthusiastic early-adopter who loves new ideas",
]
# (A) Per-line delivery → Chatterbox expressiveness knobs. exaggeration drives
# emotional intensity; lower cfg_weight slows/relaxes the pacing.
EMOTION_TTS = {
    "neutral":    {"exaggeration": 0.5, "cfg_weight": 0.5},
    "excited":    {"exaggeration": 0.8, "cfg_weight": 0.5},
    "thoughtful": {"exaggeration": 0.4, "cfg_weight": 0.3},
    "dismissive": {"exaggeration": 0.6, "cfg_weight": 0.5},
}


_kpipe = None  # Kokoro, only used to seed reference clips once
_tts = None    # Chatterbox, the actual voicing engine


def _kokoro():
    global _kpipe
    if _kpipe is None:
        from kokoro import KPipeline
        _kpipe = KPipeline(lang_code="a")  # 'a' = American English
    return _kpipe


def chatterbox():
    global _tts
    if _tts is None:
        import torch
        from chatterbox.tts import ChatterboxTTS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _tts = ChatterboxTTS.from_pretrained(device=dev)
    return _tts


def ensure_ref(voice):
    """A short Kokoro-spoken clip per voice, cached, for Chatterbox to clone."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / f"{voice}.wav"
    if not p.exists():
        audio = next(_kokoro()(REF_TEXT, voice=voice)).audio
        sf.write(p, np.asarray(audio, dtype="float32"), 24000)
    return str(p)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


RHUBARB = next(Path(__file__).resolve().parent.glob("bin/*/rhubarb"), None)


def viseme_cues(wav_path):
    """Rhubarb mouth-shape timeline for a clip → [{s, e, v}] (A–H/X). Empty if the
    binary is missing or fails, in which case the UI falls back to amplitude jaw."""
    if not RHUBARB:
        return []
    tmp = wav_path.with_suffix(".16k.wav")
    try:
        # rhubarb wants 16 kHz mono PCM
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(tmp)],
                       check=True, capture_output=True)
        out = subprocess.run([str(RHUBARB), "-r", "phonetic", "-f", "json", "--quiet", str(tmp)],
                             check=True, capture_output=True, text=True)
        return [{"s": c["start"], "e": c["end"], "v": c["value"]}
                for c in json.loads(out.stdout).get("mouthCues", [])]
    except Exception:
        return []
    finally:
        tmp.unlink(missing_ok=True)


def parse_script(text, names):
    """Pull an ordered [{speaker, line}] list out of the LLM response.
    Handles a clean JSON array, a {"...": [...]} wrapper, or ```json fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1].lstrip("json").strip()
    data = json.loads(t)
    if isinstance(data, dict):
        if "line" in data or "text" in data:   # a bare single turn
            data = [data]
        else:                                   # wrapped: {"turns": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), [])
    out = []
    for item in data:
        spk = str(item.get("speaker", "")).strip()
        line = str(item.get("line", item.get("text", ""))).strip()
        # drop "(pauses)"-style stage directions the model emits anyway; leave an
        # ellipsis so TTS renders a real pause instead of reading the word aloud
        line = re.sub(r"\s*\([^)]*\)\s*", " ... ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line == "...":
            continue
        emotion = str(item.get("emotion", "neutral")).strip().lower()
        if emotion not in EMOTION_TTS:
            emotion = "neutral"
        # snap speaker to a known name (case-insensitive), else round-robin
        match = next((n for n in names if n.lower() == spk.lower()), None)
        out.append({"speaker": match or names[len(out) % len(names)],
                    "line": line, "emotion": emotion})
    return out


def generate_script(topic, speakers):
    roster = ", ".join(f"{s['name']} ({s['persona']})" for s in speakers)
    system = (
        "You write scripts for an UNSCRIPTED-sounding podcast. Make it sound like "
        "real people talking off the cuff, NOT like an article read aloud:\n"
        "- Use contractions, casual filler and reactions: 'yeah', 'I mean', "
        "'honestly', 'right?', 'wait—', 'hold on', 'okay but'.\n"
        "- Short sentences. Let people trail off with '...' and cut in with em-dashes.\n"
        "- Each turn REACTS to what the previous speaker just said — agree, push back, "
        "build on it. No monologues, no speeches.\n"
        "- Lots of punctuation (commas, ?, —, ...) since it drives the spoken rhythm.\n"
        "- Stay fully in character. No stage directions, no narrator.\n"
        "- Tag each turn with how it's delivered via \"emotion\": one of "
        "excited, thoughtful, dismissive, neutral.\n"
        'Return ONLY a JSON array of {"speaker": "<name>", "line": "<text>", '
        '"emotion": "<delivery>"}.'
    )
    turns = max(18, len(speakers) * 7)
    user = (
        f"Topic: {topic}\nSpeakers: {roster}\n"
        f"Write a long, {turns}-turn back-and-forth where they actually talk to each "
        "other about this — interrupt, react, disagree, change their minds, and dig into "
        "specifics — in their own voices. Keep going: open it up, explore several angles, "
        "and let it build to a real conclusion. Don't wrap up early."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": {  # schema forces an array of turns, not a single object
            "type": "object",
            "properties": {"turns": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"}, "line": {"type": "string"},
                    "emotion": {"type": "string",
                                "enum": list(EMOTION_TTS)}},
                "required": ["speaker", "line", "emotion"]}}},
            "required": ["turns"],
        },
        "options": {"temperature": 0.9, "num_predict": 4096, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    return parse_script(resp["message"]["content"], [s["name"] for s in speakers])


def cleanup_old():
    """Best-effort: drop session dirs older than an hour. ponytail: good enough
    without a DB or scheduler; revisit only if this ever serves real traffic."""
    if not TMP.exists():
        return
    for d in TMP.iterdir():
        try:
            if d != REF_DIR and time.time() - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/generate")
def generate():
    topic = (request.args.get("topic") or "").strip()
    if not topic:
        return "topic required", 400

    def stream():
        sid = uuid.uuid4().hex
        outdir = TMP / sid
        outdir.mkdir(parents=True, exist_ok=True)

        # fixed order: index 0 = woman (left head), index 1 = man (right head)
        p = random.sample(PERSONAS, 2)
        speakers = [
            {"name": random.choice(FEMALE["names"]), "voice": random.choice(FEMALE["voices"]), "persona": p[0]},
            {"name": random.choice(MALE["names"]), "voice": random.choice(MALE["voices"]), "persona": p[1]},
        ]
        voice_of = {s["name"]: s["voice"] for s in speakers}
        yield sse("speakers", {"sid": sid, "speakers": speakers})

        try:
            script = generate_script(topic, speakers)
        except Exception as e:  # surface the failure to the UI instead of hanging
            yield sse("error", {"message": f"Script generation failed: {e}"})
            return
        if not script:
            yield sse("error", {"message": "Model returned an empty script."})
            return
        yield sse("script", {"lines": script})

        model = chatterbox()
        sr = model.sr
        for i, turn in enumerate(script):
            try:
                p = EMOTION_TTS[turn["emotion"]]
                wav = model.generate(
                    turn["line"], audio_prompt_path=ensure_ref(voice_of[turn["speaker"]]),
                    exaggeration=p["exaggeration"], cfg_weight=p["cfg_weight"])
                # bake the turn's trailing pause into the clip so playback and
                # export both breathe without any extra silence files
                # tight clip — the gap between turns is applied at playback/export
                # so it can be tuned live from the UI without re-voicing
                audio = np.asarray(wav.squeeze(0).cpu(), dtype="float32")
                clip = outdir / f"{i}.wav"
                sf.write(clip, audio, sr)
                yield sse("line", {"index": i, "cues": viseme_cues(clip)})
            except Exception as e:
                yield sse("error", {"message": f"TTS failed on line {i}: {e}"})
                return
        yield sse("done", {"count": len(script)})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/audio/<sid>/<int:i>.wav")
def audio(sid, i):
    return send_from_directory(TMP / sid, f"{i}.wav")


# (C) Light "recorded through a mic" chain: drop sub-bass rumble, even out levels,
# a hair of room ambience, then normalize loudness — kills the dry, sterile TTS feel.
MIC_CHAIN = "highpass=f=85,acompressor=ratio=3:attack=5:release=120,aecho=0.8:0.85:16:0.07,loudnorm=I=-16"


@app.route("/export/<sid>")
def export(sid):
    gap_ms = max(0, min(3000, request.args.get("gap", default=135, type=int)))
    outdir = TMP / sid
    clips = sorted((p for p in outdir.glob("*.wav") if p.stem.isdigit()),
                   key=lambda p: int(p.stem))
    if not clips:
        return "nothing to export", 404
    # insert the chosen inter-speaker gap (ms) between clips, matching playback
    items = [f"file '{c.name}'\n" for c in clips]
    if gap_ms:
        sr = sf.info(str(clips[0])).samplerate
        sf.write(outdir / "_pause.wav", np.zeros(int(gap_ms / 1000 * sr), "float32"), sr)
        items = [x for c in clips for x in (f"file '{c.name}'\n", "file '_pause.wav'\n")]
    (outdir / "list.txt").write_text("".join(items))
    mp3 = outdir / "podcast.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
         "-af", MIC_CHAIN, "podcast.mp3"],
        cwd=outdir, check=True, capture_output=True)
    return send_file(mp3, as_attachment=True, download_name="podcast.mp3")


if __name__ == "__main__":
    cleanup_old()
    # Warm up at boot so the model download/load happens once here, not inside the
    # first request (which would stall voicing right after the script is written).
    print("Loading TTS model and seeding voice references…")
    chatterbox()
    for v in VOICES:
        ensure_ref(v)
    print("Ready → http://localhost:5000")
    app.run(port=5000, threaded=True)
