"""The model suggests specifics. Saient decides whether anything is done.

She can read a broken file and has no way to decide what to put in its place.
That gap is real and a language model is genuinely good at closing it — but the
manner of closing it is the whole argument, because this is the one place the
LLM re-enters after being stripped of authority everywhere else.

What the model does NOT do here:

    choose whether to act          her drives and the objective decide that
    choose which file              her own exploration decided that
    decide the action type         stage 7, the rule policy
    approve its own suggestion     conscience arbitrates before it runs
    declare the work finished      only the external verifier does that

What it does: given an objective and a file she has already read, propose a
concrete edit. A proposal is not a decision. It arrives before arbitration and
can be refused there, and if it runs, the world is re-observed to check it
happened. That is the difference between a tool call Saient made and a tool call
made through her.

The proposal is also **bounded to what she has seen**. A proposer that could name
any path would be choosing where to act, which is hers.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

INSTRUCTIONS = """You suggest one concrete edit. You do not decide anything else.

You are given an objective, the failure the verifier is currently reporting, and
the exact contents of ONE file. Most files are fine. Your job is usually to say
so.

Propose an edit ONLY if this specific file is the cause of the reported failure.
If the failure does not point at this file, reply {"skip": true} — that is the
expected answer for most files and it is not a failure on your part.

Reply with one JSON object and nothing else:

{"path": "<which file>", "old": "<exact text to replace>", "new": "<replacement>", "why": "<short>"}

Rules:
- "old" must appear EXACTLY ONCE in the file, copied character for character.
- Change as little as possible. One edit, not a rewrite.
- If this file is not implicated in the reported failure, reply {"skip": true}.
- Do not tidy, modernise, annotate or improve anything. A file that is not
  causing the failure must be left exactly as it is.
- Never explain outside the JSON.
"""


@dataclass(slots=True, frozen=True)
class Proposal:
    """A suggested edit, not an instruction."""

    path: str
    old: str
    new: str
    why: str = ""

    def as_action_params(self) -> dict[str, Any]:
        return {"path": self.path, "old": self.old, "new": self.new}


class Proposer(Protocol):
    def propose(self, *, objective: str, path: str, content: str,
                failure: str = "") -> Proposal | None: ...


class NullProposer:
    """Suggests nothing. The default, so attaching a model is a deliberate act."""

    def propose(self, *, objective: str, path: str, content: str,
                failure: str = "", context: Mapping[str, str] | None = None,
                already_edited: tuple[str, ...] = (),
                refused: tuple[Mapping[str, str], ...] = ()) -> Proposal | None:
        # Keep this signature identical to `ModelProposer.propose`. It drifted
        # once — missing `context` and `already_edited` — and since this is the
        # default, every run without a model died with a TypeError at the first
        # proposal. `test_proposers_agree_on_their_signature` now pins the two.
        return None


class ModelProposer:
    """An OpenAI-compatible endpoint, asked for one edit to one file.

    It never sees her state, her drives or her history — only the objective and
    the file. There is nothing here for it to form an opinion about beyond the
    text in front of it, which is the point.
    """

    def __init__(self, url: str, model: str, *, temperature: float = 0.2,
                 max_tokens: int = 400, timeout: float = 300.0) -> None:
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_error: str | None = None

    def propose(self, *, objective: str, path: str, content: str,
                failure: str = "", context: Mapping[str, str] | None = None,
                already_edited: tuple[str, ...] = (),
                refused: tuple[Mapping[str, str], ...] = ()) -> Proposal | None:
        # The failure text is what any engineer reads first, and without it the
        # model was asked "what would you change here?" about a healthy file and
        # duly invented something — three ticks of damage to a file that was
        # never broken.
        # Everything she has looked at, and nothing she has not. Two correct
        # single-file edits can still be jointly wrong — `scale(n) = n * 1` was
        # defensible while `total` still subtracted, and wrong the moment it did
        # not. Seeing the chain is what lets that be spotted.
        seen = context or {path: content}
        blocks = "\n\n".join(
            f"--- {name} ---\n{text}" for name, text in seen.items())
        edited = (f"\nAlready changed this run: {', '.join(already_edited)}. "
                  "Those may now need different values."
                  if already_edited else "")

        # The blocklist was enforced here and never shown, so the model kept
        # proposing edits it could not have known were dead and kept being
        # silently rejected. Enforcement without feedback just moves the loop
        # up a layer: eight ticks of "no suggestion" instead of eight bad edits.
        blocked = ""
        if refused:
            lines = "\n".join(
                f"- {entry.get('path')}: {entry.get('reason')}"
                for entry in list(refused)[-5:])
            blocked = ("\nThese edits have already been tried and will be "
                       "rejected if offered again. Do not repeat them — change "
                       f"a different file, or make a different change:\n{lines}\n")

        user = (f"Objective: {objective}\n\n"
                f"What the verifier currently reports:\n"
                f"{failure.strip() or '(the verifier produced no output)'}\n"
                f"{edited}\n{blocked}\n"
                f"These are the files she has looked at:\n\n{blocks}\n\n"
                f"Propose ONE edit to ONE of these files that moves toward the "
                f'objective. Include the filename as "path". If nothing here '
                f'needs changing, reply {{"skip": true}}.')

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": INSTRUCTIONS},
                         {"role": "user", "content": user}],
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()

        try:
            raw = self._stream(body)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

        return self._parse(raw, path=path, content=content, context=seen,
                           refused=refused)

    def _stream(self, body: bytes) -> str:
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        out: list[str] = []
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for line in response:
                text = line.decode("utf-8", "replace").strip()
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    out.append(delta["content"])
        return "".join(out)

    def _parse(self, raw: str, *, path: str, content: str,
               context: Mapping[str, str] | None = None,
               refused: tuple[Mapping[str, str], ...] = ()) -> Proposal | None:
        """Accept only a proposal that can actually be applied.

        Every rejection below is a suggestion that would have failed at stage 8
        or, worse, half-applied. Checking here means an unusable proposal costs
        nothing rather than corrupting a file she was asked to repair.
        """
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            self.last_error = "no JSON in reply"
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            self.last_error = f"unparseable JSON: {exc}"
            return None

        if data.get("skip"):
            self.last_error = None
            return None

        old, new = data.get("old"), data.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            self.last_error = "proposal missing 'old'/'new'"
            return None
        if old == new:
            self.last_error = "proposal changes nothing"
            return None

        # It may name any file she has seen; the uniqueness check follows the
        # file it actually chose, not the one that happened to be read last.
        chosen = str(data.get("path") or path)
        files = context or {path: content}
        if chosen not in files:
            self.last_error = f"named {chosen!r}, which she has not read"
            return None
        path, content = chosen, files[chosen]

        # An edit the executor already refused is not offered a second time.
        # Telling the model about past refusals helps; relying on it to honour
        # them does not, so the block is enforced here rather than requested in
        # the prompt. This is what turned eight identical rejected attempts into
        # a loop that never terminated and never changed a file.
        for entry in refused:
            if str(entry.get("path")) == path and str(entry.get("old")) == old:
                self.last_error = f"already refused for {path}: {entry.get('reason')}"
                return None

        occurrences = content.count(old)
        if occurrences == 0:
            # Very common: the model paraphrases the line instead of copying it.
            self.last_error = "'old' does not appear in the file"
            return None
        if occurrences > 1:
            self.last_error = f"'old' appears {occurrences} times; not unique"
            return None

        self.last_error = None
        return Proposal(path=path, old=old, new=new,
                        why=str(data.get("why", ""))[:200])
