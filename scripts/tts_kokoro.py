#!/usr/bin/env python3
"""Kokoro TTS for Saient.
Reads JSON from stdin, writes WAV as base64 to stdout.
Progress lines ({"progress": 0-100}) go to stderr.
"""
import base64, io, json, os, sys

# Force CPU if GPU is nearly full — TTS is fast on CPU anyway
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Runtime synthesis is local-only. Full Setup prefetches the model, voices and
# language model; a missing asset must be reported instead of quietly fetching.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# stdout is our JSON channel and must stay clean. Third-party code writes to it:
# misaki/en.py calls spacy.cli.download(), which spawns pip as a SUBPROCESS.
# A subprocess inherits OS fd 1, so redirect_stdout / reassigning sys.stdout would
# not catch it — this has to happen at the file-descriptor level.
_REAL_OUT = os.dup(1)   # private handle to the true stdout
os.dup2(2, 1)           # fd 1 → stderr, inherited by every child process


def _emit(obj):
    """Write our one JSON result to the real stdout, bypassing the redirect."""
    os.write(_REAL_OUT, json.dumps(obj).encode())


def main():
    payload = json.loads(sys.stdin.read())
    text     = payload.get("text", "").strip()
    voice    = payload.get("voice", "af_heart")
    speed    = float(payload.get("speed", 1.0))
    lang     = payload.get("lang", "a")   # 'a'=American, 'b'=British

    if not text:
        raise RuntimeError("No text provided")

    print(json.dumps({"progress": 10}), file=sys.stderr, flush=True)

    if lang in ("a", "b"):
        try:
            import en_core_web_sm  # noqa: F401 - availability check only
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro's English language data is not installed. Run Full Setup while temporary Internet access is authorized."
            ) from exc

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

    _emit({
        "base64_wav": b64,
        "duration":   round(len(audio) / 24000, 2),
        "sample_rate": 24000,
    })
    print(json.dumps({"progress": 100}), file=sys.stderr, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # must also go to the real stdout — sys.stdout now points at stderr
        _emit({"error": str(e)})
        sys.exit(1)
