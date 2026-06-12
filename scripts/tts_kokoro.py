#!/usr/bin/env python3
"""Kokoro TTS for Saient.
Reads JSON from stdin, writes WAV as base64 to stdout.
Progress lines ({"progress": 0-100}) go to stderr.
"""
import base64, io, json, os, sys

# Force CPU if GPU is nearly full — TTS is fast on CPU anyway
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def main():
    payload = json.loads(sys.stdin.read())
    text     = payload.get("text", "").strip()
    voice    = payload.get("voice", "af_heart")
    speed    = float(payload.get("speed", 1.0))
    lang     = payload.get("lang", "a")   # 'a'=American, 'b'=British

    if not text:
        raise RuntimeError("No text provided")

    print(json.dumps({"progress": 10}), file=sys.stderr, flush=True)

    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np

    print(json.dumps({"progress": 30}), file=sys.stderr, flush=True)

    pipe = KPipeline(lang_code=lang, repo_id='hexgrad/Kokoro-82M')
    gen  = pipe(text, voice=voice, speed=speed)

    print(json.dumps({"progress": 60}), file=sys.stderr, flush=True)

    chunks = [audio for _, _, audio in gen]
    if not chunks:
        raise RuntimeError("No audio generated")
    audio = np.concatenate(chunks)

    print(json.dumps({"progress": 90}), file=sys.stderr, flush=True)

    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    sys.stdout.write(json.dumps({
        "base64_wav": b64,
        "duration":   round(len(audio) / 24000, 2),
        "sample_rate": 24000,
    }))
    sys.stdout.flush()
    print(json.dumps({"progress": 100}), file=sys.stderr, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e)}))
        sys.stdout.flush()
        sys.exit(1)
