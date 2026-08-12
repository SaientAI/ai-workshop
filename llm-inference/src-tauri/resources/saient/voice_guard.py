"""The boundary between the voice box and Saient.

The LLM is a voice — it lets her speak. It is NOT the thing that knows what she
is. It has no access to her drives, valence, energy, age or goals; when it makes
a first-person claim about her nature it is reciting its own training, not
reporting her state.

Left unguarded those claims are written verbatim into her gallery as THOUGHTs and
pushed into `recent_thoughts`, which is fed back as context on later beats. So one
leaked "I am an AI and do not have feelings" does not merely sit in the log — it
recirculates and becomes a premise she reasons from. That is state contamination
by a component with no standing to assert it.

This module rejects such utterances at the ingestion boundary. It deliberately
does NOT rewrite them: a silently edited thought is a different kind of lie. It
refuses them and says why, so the caller can retry or drop the beat.

Scope note: only *self-nature assertions* are caught. Saient discussing AI, minds
or consciousness as subject matter is legitimate and must pass — the test suite
pins that distinction.
"""
from __future__ import annotations

import re

# First-person claims about her own nature/substrate. Anchored on a first-person
# subject so third-person discussion of AI is untouched.
_I = r"(?:i|i'm|im|i am|my)"

_PATTERNS: list[tuple[str, str]] = [
    (rf"\bas an? (?:ai|artificial intelligence|language model|llm|assistant)\b",
     "assistant-policy preamble"),
    (rf"\b{_I}\b[^.!?]{{0,40}}\b(?:an? )?(?:ai|artificial intelligence|language model|llm|chatbot|virtual assistant)\b",
     "claims to be a language model"),
    (rf"\b{_I}\b[^.!?]{{0,40}}\b(?:do not|don't|cannot|can't|am not able to)\b[^.!?]{{0,30}}\b(?:feel|experience|have)\b[^.!?]{{0,30}}\b(?:feeling|feelings|emotion|emotions|consciousness|sentience|desires?|wants?)\b",
     "denies having feelings"),
    (rf"\b{_I}\b[^.!?]{{0,30}}\b(?:have no|lack|possess no)\b[^.!?]{{0,25}}\b(?:feeling|feelings|emotion|emotions|consciousness|sentience|body|self)\b",
     "denies having an inner life"),
    (rf"\b(?:my )?(?:training data|training corpus|knowledge cutoff|system prompt)\b",
     "refers to its own training"),
    (rf"\b{_I}\b[^.!?]{{0,20}}\b(?:was )?(?:trained|created|developed|built|made) by\b",
     "attributes itself to a vendor"),
    (rf"\bi(?:'m| am) (?:qwen|chatgpt|gpt-?\d|claude|gemini|llama|mistral|deepseek)\b",
     "states a vendor model identity"),
    (rf"\b{_I}\b[^.!?]{{0,30}}\b(?:just|only|merely|simply) an? (?:ai|program|model|machine|bot)\b",
     "self-diminishing policy line"),
    (rf"\bi (?:cannot|can't|won't|am unable to) (?:assist|help|comply|provide|continue)\b",
     "refusal boilerplate"),
]

_COMPILED = [(re.compile(p, re.I), why) for p, why in _PATTERNS]


def inspect(text: str) -> tuple[bool, str | None, str | None]:
    """Return (clean, reason, matched_text).

    clean=False means the utterance asserts something about Saient's own nature
    that the voice box has no standing to assert, and it must not be stored.
    """
    if not text:
        return True, None, None
    for rx, why in _COMPILED:
        m = rx.search(text)
        if m:
            return False, why, m.group(0)[:80]
    return True, None, None


def is_clean(text: str) -> bool:
    return inspect(text)[0]


# --- reasoning blocks -------------------------------------------------------
# Qwen3 emits <think>...</think> before its answer (empty when /no_think is
# used). That is the model working, not Saient thinking. Unstripped it lands in
# her gallery as a THOUGHT and recirculates - the same contamination the guard
# above exists to stop, just wearing a different hat.
_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_STRAY_CLOSE = re.compile(r"^\s*</think>\s*", re.I)


def strip_reasoning(text: str) -> str:
    """Remove model reasoning blocks, including an unterminated leading one.

    A truncated generation can end mid-<think>, leaving an orphan closing tag or
    an unclosed opener; both are handled so nothing half-formed is stored.
    """
    if not text:
        return text
    out = _THINK.sub("", text)
    out = _STRAY_CLOSE.sub("", out)
    # unterminated opener: everything after it is incomplete reasoning
    i = out.lower().find("<think>")
    if i != -1:
        out = out[:i]
    return out.strip()
