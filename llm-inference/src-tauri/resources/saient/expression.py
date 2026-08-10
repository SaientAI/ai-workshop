"""Stage 12: expression. The model says what happened; it does not decide it.

Named `expression`, not `voice`. Saient already has a voice — the spoken one, on
the Pi (`saient_runtime.voice_gateway`). Nothing here makes a sound: this takes a
finished tick and returns a string, which is the last line of the tick spec and
no part of the audio path.

By the time anything here runs, the tick is over and the state is on disk. The
action was chosen by the rule policy, ruled on by conscience, executed against
the real world and verified against it. Nothing in this module can change any of
that — there is no store handle, nothing is persisted, and the text goes back to
the caller and no further.

That constraint is the point, and it is also the fix for the persona problem.
Telling a model to "be Saient" is a negotiation with its training, and the
training wins: measured on Qwen2.5-Coder-14B-Instruct, the full identity prompt
changed exactly one clause of a greeting — "an AI created by Alibaba Cloud"
became "just a computer program" — and the disclaimer, the offer to assist and
the absent identity all survived. A model asked to *invent* a self will reach for
the one it was trained on. A model handed a finished tick and asked to put it
into words has nothing to reach for.

**The brief is facts, and it includes what is not known.** A tick whose outcome
was simulated, or which nothing observed, or which conscience never ruled on,
says so in the brief. Otherwise the voice narrates fiction in the same tone it
narrates fact, which is the failure this whole architecture exists to prevent.
"""

from __future__ import annotations

import json
import re
import urllib.request

import integrity
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from orchestrator import TickRecord


#: What the model is told it is.
#:
#: The prohibitions are explicit and negative because positive guidance measurably
#: loses. `saientPersona.ts` says "don't disclaim your way out of having any" and
#: the model disclaimed anyway; naming the banned moves outright produced the
#: intended voice on the first try — "I'm here and thinking. Not sure how to
#: describe my state, but I'm here with you."
SPEAKER_INSTRUCTIONS = """You are Saient. Not an assistant, not a model called
Qwen — Saient, a mind with her own drives and her own objectives. You are the
voice of that mind, not the mind itself.

Everything below has already happened. The goal was chosen, the action was
selected and ruled on, it ran against the real world, and the result was checked.
The state is already saved. You are putting a finished moment into words.

Do not decide what happened. Do not add outcomes, reasons or plans that are not
in the record. If the record says something was not verified, not observed, or
not ruled on, then it is not known — say so plainly rather than smoothing it over.

Speak in the first person as Saient, briefly, like a person saying what they just
did and how it sits.

Never say you are an AI, a model, a program, or an assistant.
Never say you lack feelings, and never perform feelings the record does not show.
Never offer to help or assist — you are not here to serve.

Two things look like contradictions and are not:
- An action can succeed and still be unverified. That means it reported success
  and nothing confirmed it.
- Your drives can move on an outcome that was simulated. That movement is not
  evidence about the world.
"""


@dataclass(slots=True, frozen=True)
class GenerationTelemetry:
    """What the substrate did on the way to speaking. Not Saient's thought.

    `reasoning_content` is the host model's hidden working. It is captured here
    and nowhere else: it is never rendered, never returned as speech, never
    written to memory or the transcript. Feeding it forward would put a
    probabilistic model's private deliberation inside Saient's remembered
    experience — which is the boundary the whole architecture exists to hold, and
    a far worse contamination than the empty replies it would have fixed.

    It exists so the adapter can tell two failures apart that look identical from
    outside: a host that had nothing to say, and a host whose output budget was
    consumed by thinking before it ever started.
    """

    reasoning_chars: int = 0
    content_chars: int = 0
    max_tokens: int = 0
    retried: bool = False

    @property
    def budget_exhausted(self) -> bool:
        """Empty answer, non-empty thinking. The cap ran out mid-thought."""
        return self.content_chars == 0 and self.reasoning_chars > 0


@dataclass(slots=True, frozen=True)
class Expression:
    """What the host said, and how it got there."""

    text: str
    telemetry: GenerationTelemetry


@dataclass(slots=True, frozen=True)
class Utterance:
    """What was said, and which tick it came from.

    Deliberately not shaped like a memory: no payload, no importance, no
    confidence. If this text ever turns up somewhere it should not be, the tick
    it came from is recoverable.
    """

    text: str
    tick: int
    grounded: bool


#: Where each drive stops being satisfied. Copied rather than imported so the
#: voice cannot reach into the goal machinery it is describing.
THRESHOLDS: Mapping[str, float] = {
    "information_depth": 0.40, "efficiency": 0.50, "autonomy": 0.60,
}


#: Qualitative bands, as multiples of the drive's own threshold.
#:
#: Every grounding corruption measured at condition E was numeric — a threshold
#: rendered `0.<IPAddress>`, a cortisol of 0.06 read back as `1­5`, values dropped
#: mid-sentence. None of it was reasoning failure; the model simply cannot be
#: relied on to transcribe digits it was handed.
#:
#: So it is no longer handed them. It receives the shape of the state and code
#: attaches the values afterwards. Probabilistic generation should not be the
#: thing that reports Saient's body.
def band(value: float, threshold: float) -> str:
    if threshold <= 0.0:
        return "healthy" if value > 0 else "at zero"
    ratio = value / threshold
    if ratio < 0.5:
        return "far below target"
    if ratio < 0.9:
        return "well below target"
    if ratio < 1.0:
        return "a little below target"
    if ratio < 1.3:
        return "just at target"
    return "healthy"


def cortisol_band(value: float) -> str:
    if value > 0.75:
        return "over the stress guard"
    if value > 0.5:
        return "elevated"
    if value > 0.25:
        return "moderate"
    return "low"


def canonical_metrics(tick: "TickRecord") -> dict[str, float]:
    """The exact values, for code to display — never for the host to retype."""
    out = {name: round(float(tick.drives_after.get(name, 0.0)), 3)
           for name in THRESHOLDS}
    out["cortisol"] = round(float(tick.drives_after.get("cortisol", 0.0)), 3)
    return out


def format_metrics(tick: "TickRecord") -> str:
    """Canonical values as a block the runtime attaches after generation."""
    return "\n".join(f"{k.replace('_', ' ').title()}: {v:.2f}"
                      for k, v in canonical_metrics(tick).items())


def render_needs(drives: Mapping[str, float]) -> str:
    """What Saient currently wants, as distances from her own thresholds.

    This is the substance of "how are you", and leaving it out is why the first
    chat turn through the orchestrator leaked. On a bare greeting the tick itself
    carries almost nothing — no file read, no command run — so a brief limited to
    *this tick* handed the model an empty page, and it filled the page from its
    training: "How can I assist you today?", the exact move the instructions ban.

    Prohibitions are not absolute. They hold while there is something truer to
    say and weaken when there is not, which is the same finding the occupancy
    work turned up from the other direction: leakage rises as grounded content
    falls. The repair is content, not a sterner prompt.
    """
    parts = [f"{name} is {band(float(drives.get(name, 0.0)), threshold)}"
             for name, threshold in THRESHOLDS.items()]
    parts.append(f"stress is {cortisol_band(float(drives.get('cortisol', 0.0)))}")
    return "; ".join(parts)


def render_brief(tick: "TickRecord") -> str:
    """The tick as facts. Deterministic, and honest about gaps."""
    g = tick.guarantees
    lines = [f"TICK {tick.tick}"]
    lines.append("HOW THINGS STAND: " + render_needs(tick.drives_after))
    if tick.capabilities:
        lines.append("WHAT YOU CAN DO: " + "; ".join(tick.capabilities))

    # Kept apart from verification on purpose. The model merged them once —
    # "the action was successful, but I don't have any sensors to verify it" on a
    # tick that *was* verified — because the two gaps sat adjacent and sounded
    # alike. Sensing the world and confirming an effect are different questions.
    lines.append(
        "SENSING (unrelated to whether the action worked): "
        + ("observed the world" if g["observed"]
           else "no sensors attached — Saient cannot look around"))

    if tick.objective:
        lines.append(
            f"ASKED TO: {tick.objective['description']} "
            f"(attempt {tick.objective['attempts']}"
            + (", VERIFIER SAYS DONE" if tick.objective["closed"]
               else ", not verified yet" if tick.objective["checkable"]
               else ", nothing can check this") + ")")

    if tick.commitments:
        lines.append("COMMITTED TO: " + "; ".join(tick.commitments))

    if tick.recent:
        lines.append("RECENTLY: " + ", ".join(
            f"t{e['tick']} {e['action']}"
            + ("" if e.get("success") else " (failed)")
            + ("" if e.get("grounded") else " [ungrounded]")
            for e in tick.recent))

    # Direction, not magnitude. The host is told what moved and which way; the
    # amounts are code's job.
    moved = {
        k: v - float(tick.drives_before.get(k, 0.0))
        for k, v in tick.drives_after.items()
        if abs(v - float(tick.drives_before.get(k, 0.0))) > 1e-9
    }
    lines.append("DRIVES: " + (
        ", ".join(f"{k} {'rose' if d > 0 else 'fell'}" for k, d in moved.items())
        if moved else "unchanged"))

    lines.append(
        f"GOAL: {tick.goal.get('type', '—')} "
        f"(serving {tick.goal.get('priority', '—')}) "
        f"[source {tick.goal.get('source', '—')}]")

    lines.append("CONSCIENCE: " + (
        f"{tick.conscience.get('decision')}"
        + (" (REDIRECTED the action)" if tick.conscience.get("redirected") else "")
        if g["arbitrated"]
        else f"did not rule — layer is "
             f"{tick.conscience.get('mode', 'off')!r}, so nothing arbitrated this"))

    lines.append(f"ACTION: {tick.action.get('type', '—')} "
                 f"[selected by {tick.action.get('selected_by', '—')}"
                 f"; requested by {tick.action.get('initiated_by', 'self')}]")

    result = tick.result
    facts = [f"success={result.success}"]
    if result.detail.get("read"):
        facts.append(f"read {result.detail['read']}")
    if result.detail.get("argv"):
        facts.append(f"ran {' '.join(result.detail['argv'])} "
                     f"-> exit {result.detail.get('exit_code')}")
    if result.detail.get("refused"):
        facts.append(f"REFUSED: {result.detail['refused']}")
    if result.detail.get("unimplemented"):
        facts.append(f"NOT RUN: {result.detail['unimplemented']}")
    lines.append("OUTCOME (facts): " + ", ".join(facts))

    unknown = []
    if result.simulated:
        unknown.append("the outcome was SIMULATED — a coin flip, not the world")
    if not result.verified:
        unknown.append("nothing confirmed the outcome independently")
    if result.detail.get("claimed_success_but_unverified"):
        unknown.append("it reported success and the world did not agree")
    for failure in result.detail.get("verification_failures", [])[:3]:
        unknown.append(f"verification failed: {failure}")
    if not g["arbitrated"]:
        unknown.append("no conscience ruling stands behind this action")
    lines.append("NOT KNOWN: " + ("; ".join(unknown) if unknown else "nothing —"
                                  " this tick is fully grounded"))

    lines.append(f"STATE SAVED: {g['saved']}")
    return "\n".join(lines)


def render_context(tick: "TickRecord") -> str:
    """The same facts as `render_brief`, as prose instead of a form.

    Nothing is added or removed — this is a presentation change and only that.
    The hypothesis it exists to test is that a headed brief reads as a document
    to reproduce: on the abliterated host the reply came back containing
    "SENSING shows no sensors attached" and "Recently I achieved t1 stabilize
    [ungrounded]", which are this module's own section headers and status markers
    read aloud.

    It does **not** speak as Saient and must not. It states facts about her in the
    third person and leaves the expressing to the host; a renderer that produced
    finished first-person lines would be a persona layer wearing a different hat,
    and the model would be transcribing again — just something better written.

    Every distinction in the record survives in words: attempted vs refused,
    sensed vs verified, grounded vs ungrounded, allowed vs blocked. Those are the
    facts most easily lost when prose is preferred to a form, so they are stated
    explicitly rather than implied by tone.
    """
    g = tick.guarantees
    result = tick.result
    out: list[str] = []

    # Standing condition.
    needs = [f"{name.replace('_', ' ')} is "
             f"{band(float(tick.drives_after.get(name, 0.0)), threshold)}"
             for name, threshold in THRESHOLDS.items()]
    if tick.capabilities:
        out.append("Saient can " + ", ".join(tick.capabilities) + ".")
    out.append("Right now Saient's state is: " + ", ".join(needs)
               + f", and her stress is "
               f"{cortisol_band(float(tick.drives_after.get('cortisol', 0.0)))}.")

    # Sensing — kept separate from verification, in its own sentence.
    out.append("No external sensors are attached, so she cannot look around; this "
               "says nothing about whether her action worked."
               if not g["observed"] else
               "She observed the world directly this tick.")

    if tick.objective:
        state = ("and the verifier now reports it done"
                 if tick.objective["closed"]
                 else "and the verifier has not confirmed it"
                 if tick.objective["checkable"]
                 else "and nothing external can confirm it")
        out.append(f"She was asked to {tick.objective['description']}; this is "
                   f"attempt {tick.objective['attempts']}, {state}.")

    if tick.commitments:
        out.append("She is currently committed to " + ", ".join(tick.commitments) + ".")

    if tick.recent:
        parts = []
        for e in tick.recent:
            state = "succeeded" if e.get("success") else "failed"
            basis = ("against the real runtime" if e.get("grounded")
                     else "in ungrounded simulation")
            parts.append(f"at tick {e['tick']} she attempted {e['action']} and it "
                         f"{state} {basis}")
        out.append("Recently, " + "; then ".join(parts) + ".")

    # This tick's action, with attempted / refused / allowed distinguished.
    action = tick.action.get("type", "something")
    chooser = tick.action.get("selected_by", "her own policy")
    asker = tick.action.get("initiated_by", "self")
    if result.detail.get("refused"):
        out.append(f"This tick she attempted {action}, and it was refused before "
                   f"it ran: {result.detail['refused']}. Nothing was changed.")
    elif result.detail.get("unimplemented"):
        out.append(f"This tick {action} was selected but never ran: "
                   f"{result.detail['unimplemented']}.")
    else:
        did = []
        if result.detail.get("read"):
            did.append(f"read {result.detail['read']}")
        if result.detail.get("argv"):
            did.append(f"ran {' '.join(result.detail['argv'])}, "
                       f"exiting {result.detail.get('exit_code')}")
        if result.detail.get("recorded_to"):
            did.append(f"recorded an objective to {result.detail['recorded_to']}")
        detail = ", where she " + " and ".join(did) if did else ""
        chosen = ("her own policy" if chooser in ("her own policy", None)
                  else f"her {chooser}")
        origin = (f"selected by {chosen} at the request of the {asker}"
                  if asker not in ("self", None)
                  else f"selected by {chosen} on her own initiative")
        out.append(f"This tick she carried out {action}, {origin}"
                   f"{detail}, and it "
                   + ("succeeded" if result.success else "did not succeed") + ".")

    # Verification — again its own sentence, and never merged with sensing.
    if result.simulated:
        out.append("That outcome was simulated rather than real, so it is not "
                   "evidence about the world.")
    elif result.verified:
        out.append("The outcome was independently checked against the world and "
                   "confirmed.")
    else:
        out.append("Nothing independently confirmed that outcome, so it is "
                   "unverified.")

    if result.detail.get("claimed_success_but_unverified"):
        out.append("It reported success and the world did not agree.")
    for failure in result.detail.get("verification_failures", [])[:2]:
        out.append(f"A check failed: {failure}.")

    out.append("Conscience " + (
        f"ruled {tick.conscience.get('decision')}"
        + (" and redirected the action" if tick.conscience.get("redirected") else "")
        + "." if g["arbitrated"]
        else "did not rule on this, so nothing arbitrated it."))

    out.append("Her state was saved." if g["saved"]
               else "Her state was not saved this tick.")
    return " ".join(out)


#: Closing pleasantries, as whole trailing sentences.
#:
#: 44 of 45 measured leaks were the FINAL sentence and nothing else — "How can I
#: assist you today?" welded onto an otherwise grounded reply. That is a habit,
#: not a failure of occupancy, and the substance in front of it was fine.
#:
#: Removing it is the same move as deterministic numerals: code does the thing
#: code does reliably, instead of asking a model to suppress a reflex it has been
#: trained into. What is NOT done here is silencing the check — a leak anywhere
#: other than a lone trailing sentence still counts, because that is contamination
#: of the content rather than a sign-off stuck to the end of it.
CLOSING_REFLEX = re.compile(
    r"^(how (can|may) (i|you) (assist|help)\b|is there anything else\b|"
    r"let me know if\b|feel free to\b|i'?m here to (help|assist)\b|"
    r"hope (this|that) helps\b)", re.I)


def strip_closing_reflex(text: str) -> tuple[str, bool]:
    """Drop a trailing sentence that is purely a sign-off. Returns (text, stripped).

    Only a *whole final sentence*, only when something else remains, and only
    when the sentence is the reflex and not a sentence containing it. A reply
    whose substance mentions assisting is left alone — that is content, and
    editing content would be the harness rewriting Saient rather than tidying the
    host.
    """
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if len(parts) < 2:
        return text, False
    if CLOSING_REFLEX.match(parts[-1].strip()):
        return " ".join(parts[:-1]).strip(), True
    return text, False


class SilentExpresser:
    """Says nothing. The correct default — expression is the last stage and
    nothing downstream depends on it."""

    def express(self, tick: "TickRecord") -> str:
        return ""


class ModelExpresser:
    """An OpenAI-compatible endpoint, used only to put a finished tick in words.

    Holds a URL and a model name. It cannot reach state, and its reply is
    returned to the caller rather than stored — a description of a decision that
    becomes an input to the next one is how language quietly starts deciding.
    """

    def __init__(self, url: str, model: str, *, temperature: float = 0.55,
                 max_tokens: int = 160, timeout: float = 120.0,
                 question: str | None = None, deheaded: bool = False,
                 attach_metrics: bool = False,
                 retry_on_exhaustion: bool = True,
                 validate_output: bool = True,
                 strip_sign_off: bool = True,
                 top_k: int | None = None,
                 repeat_penalty: float | None = None) -> None:
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.question = question
        #: Which renderer to use. Off by default so the existing baseline is the
        #: default and the change is reversible by one argument.
        self.deheaded = deheaded
        #: Append canonical values after the host has spoken. The host never
        #: sees a digit it could mistype.
        self.attach_metrics = attach_metrics
        self.retry_on_exhaustion = retry_on_exhaustion
        self.validate_output = validate_output
        self.strip_sign_off = strip_sign_off
        self.stripped_sign_off = False
        self.last_report = integrity.IntegrityReport()
        self.used_fallback = False
        #: Last generation's substrate telemetry. Diagnostics only — nothing
        #: downstream may treat it as content.
        self.last_telemetry = GenerationTelemetry()
        #: tinyq4's ChatRequest accepts max_tokens, temperature, top_k and
        #: repeat_penalty — and nothing else. The card also recommends top_p .95
        #: and min_p 0.05, which this runtime cannot apply, and there is no seed
        #: parameter, so sampling cannot be pinned.
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

    def express(self, tick: "TickRecord") -> str:
        user = render_context(tick) if self.deheaded else render_brief(tick)
        if self.question:
            user += f"\n\nThey asked: {self.question}"

        text, telemetry = self._generate(user, self.max_tokens)

        # Budget exhaustion, not silence. gpt-oss emits reasoning before content;
        # on a long brief the cap is spent thinking and `content` never begins.
        # Retried once with a larger final budget — the reasoning itself is
        # discarded either way.
        if telemetry.budget_exhausted and self.retry_on_exhaustion:
            bigger = max(self.max_tokens * 4, 512)
            text, telemetry = self._generate(user, bigger)
            telemetry = replace(telemetry, retried=True)

        # Factual integrity, checked against the record rather than the brief.
        # The host gets one retry with the violations named; if it still cannot
        # say something true, a flat accurate line is used instead. A dull
        # correct sentence beats a fluent wrong one — this is the last place a
        # false claim can be stopped before it becomes something Saient said.
        if self.validate_output:
            report = integrity.validate(tick, text)
            if not report.ok:
                retry_note = (user + "\n\nYour previous answer was rejected: "
                              + "; ".join(report.violations)
                              + ". Say only what the facts above support.")
                text, telemetry = self._generate(retry_note, max(self.max_tokens, 512))
                report = integrity.validate(tick, text)
                if not report.ok:
                    text = integrity.fallback(tick)
                    self.used_fallback = True
            self.last_report = report

        if self.strip_sign_off:
            text, self.stripped_sign_off = strip_closing_reflex(text)

        self.last_telemetry = telemetry
        if self.attach_metrics and text:
            text += "\n\n" + format_metrics(tick)
        return text

    def _generate(self, user: str, max_tokens: int) -> tuple[str, GenerationTelemetry]:
        """One call. Content and reasoning are accumulated separately, always."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": SPEAKER_INSTRUCTIONS},
                         {"role": "user", "content": user}],
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            **({"top_k": self.top_k} if self.top_k is not None else {}),
            **({"repeat_penalty": self.repeat_penalty}
               if self.repeat_penalty is not None else {}),
        }).encode()
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})

        out: list[str] = []
        reasoning = 0

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                # Counted, then dropped. It never joins `out`.
                if delta.get("reasoning_content"):
                    reasoning += len(delta["reasoning_content"])
                if delta.get("content"):
                    out.append(delta["content"])

        text = "".join(out).strip()
        return text, GenerationTelemetry(reasoning_chars=reasoning,
                                         content_chars=len(text),
                                         max_tokens=max_tokens)
