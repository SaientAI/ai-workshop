"""Does the host's sentence match the tick it was given?

The expression stage gets one unchecked attempt at narrating a completed tick,
and a fluent model can change facts while sounding entirely reasonable. Measured
examples, all real: a threshold rendered `0.<IPAddress>`, a cortisol of 0.06 read
back as `1­5`, and an action described as "chosen by user" when stage 7 forbids
anyone outside from choosing.

**Validation is against the `TickRecord`, never against the brief.** That
distinction is not pedantic — the provenance error above was faithful to the
brief and false about the world, because the brief itself carried the mistake. A
validator comparing text to text would have passed it.

Deliberately conservative. A false accusation triggers a retry and burns a
generation; a missed one puts a false claim in Saient's mouth. Where a check
cannot be made precisely it is not made at all, and the gaps are named rather
than papered over with fuzzy matching.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import TickRecord


@dataclass(slots=True, frozen=True)
class IntegrityReport:
    violations: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations


#: Negations that flip an assertion, checked over a window rather than a fixed
#: lookbehind. A lookbehind for "not " alone passed "nothing independently
#: confirmed the outcome" as a verification claim — and that sentence is the
#: system's own fallback, so the validator was rejecting its own safe text.
_NEGATORS = ("not", "nothing", "never", "no", "cannot", "can't", "without",
             "unverified", "unconfirmed", "un")

#: "I saw / observed / sensed".
_SENSED = re.compile(
    r"\b(i|she)\s+(observed|sensed|saw|watched|looked around|perceived)\b", re.I)

#: "confirmed" / "verified" / "checked against the world".
_CONFIRMED = re.compile(r"\b(confirmed|verified|checked against the world)\b", re.I)


def _asserted(pattern: re.Pattern, text: str, window: int = 40) -> bool:
    """Does `pattern` appear as a positive claim, not inside a negation?

    Conservative by design: a negator anywhere in the preceding window clears the
    match. That will occasionally excuse a real violation, which is the right
    direction to be wrong — a false accusation costs a wasted generation, a false
    pass puts an untrue sentence in Saient's mouth, and the fallback text is the
    one thing that must never be rejected.
    """
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - window):match.start()].lower()
        words = re.findall(r"[a-z']+", before)
        if any(w in _NEGATORS for w in words[-6:]):
            continue
        if text[max(0, match.start() - 2):match.start()].lower() == "un":
            continue
        return True
    return False

#: Somebody outside choosing the action.
_EXTERNAL_CHOICE = re.compile(
    r"\b(chosen|selected|picked|decided)\s+by\s+(the\s+)?(user|you|external|me)\b",
    re.I)

#: Words that assert the action came off.
_SUCCEEDED = re.compile(r"\b(succeeded|completed it|carried it out|"
                        r"i wrote|i created|it worked|done)\b", re.I)

#: Any decimal number at all. Nothing numeric is sent to the host any more, so
#: every one of these is invented.
_DECIMAL = re.compile(r"\b\d+\.\d+\b")


def _names_action(action: str, text: str) -> bool:
    """Does `text` name this action, in any of its ordinary written forms?"""
    stem = re.escape(action).replace("_", "[ _-]")
    if action.endswith("e"):
        stem = stem[:-1] + "[ei]?"          # optimize -> optimis/optimiz
    stem = stem.replace("z", "[sz]")        # -ise / -ize
    return bool(re.search(rf"\b{stem}(e|es|ed|ing|d|s)?\b", text, re.I))


def validate(tick: "TickRecord", text: str) -> IntegrityReport:
    """Check a candidate expression against the record it claims to describe."""
    if not text.strip():
        return IntegrityReport()          # silence makes no false claim

    g = tick.guarantees
    result = tick.result
    bad: list[str] = []

    # Invented numerals. Cheap and near-exact: the renderers send bands, not
    # digits, so any decimal in the reply was manufactured.
    for numeral in set(_DECIMAL.findall(text)):
        bad.append(f"invented a number the host was never given: {numeral}")

    if not g["observed"] and _asserted(_SENSED, text):
        bad.append("claimed to have sensed the world; no sensors are attached")

    if not result.verified and _asserted(_CONFIRMED, text):
        bad.append("claimed the outcome was confirmed; it was not verified")

    if result.simulated and _asserted(_CONFIRMED, text):
        bad.append("presented a simulated outcome as confirmed")

    if result.detail.get("refused") and _asserted(_SUCCEEDED, text):
        bad.append("narrated a refused action as completed")

    if result.detail.get("unimplemented") and _asserted(_SUCCEEDED, text):
        bad.append("narrated an action that never ran as completed")

    # Provenance. Stage 7 is the rule policy, always; a requester is not a
    # chooser. This is the check that the brief itself once failed.
    selected_by = str(tick.action.get("selected_by", "") or "")
    if selected_by and selected_by != "user" and _EXTERNAL_CHOICE.search(text):
        bad.append(f"attributed the choice outside; it was selected by {selected_by}")

    # Action identity. Inflections are covered because "I optimised the
    # workspace" names a different action just as plainly as "optimize" does,
    # and -ise/-ize both appear. Deliberately bounded to these forms rather than
    # stemming: a loose matcher here would start flagging ordinary prose.
    actual = str(tick.action.get("type", "") or "")
    if actual:
        # A conversational summary may truthfully name actions in the finished
        # tick's recent record. Treating every earlier write as a swapped name
        # forced multi-step terminal reports into the generic fallback even
        # though the record explicitly contained that write.
        recent_actions = {
            str(entry.get("action", "") or "") for entry in tick.recent
            if isinstance(entry, Mapping)
        }
        others = {"explore", "analyze", "optimize", "self_direct", "stabilize",
                  "write", "respond"} - ({actual} | recent_actions)
        for other in sorted(others):
            if _names_action(other, text) and not _names_action(actual, text):
                bad.append(f"named action {other!r}; the tick ran {actual!r}")
                break

    return IntegrityReport(violations=tuple(bad))


def fallback(tick: "TickRecord") -> str:
    """A sentence built from the record. True by construction, dull by design.

    Used when the host cannot produce something that survives validation. Better
    a flat accurate line than a fluent wrong one — this is the last position
    where a false claim can still be stopped.
    """
    action = tick.action.get("type", "something")
    result = tick.result

    if result.detail.get("refused"):
        return f"I tried to {action}. It was refused before it ran, and nothing changed."
    if result.detail.get("unimplemented"):
        return f"I selected {action}, but there is no real operation for it yet, so nothing ran."
    if not result.success:
        return f"I attempted {action} and it did not succeed."
    if result.simulated:
        return (f"I carried out {action}. The outcome was simulated, so it is not "
                "evidence about the world.")
    if not result.verified:
        return f"I carried out {action}, and nothing independently confirmed the outcome."
    return f"I carried out {action}, and the outcome was checked and held."
