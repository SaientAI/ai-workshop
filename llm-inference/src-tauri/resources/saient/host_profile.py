"""Learning the interface a host needs, rather than asking it to change.

The question this answers is not "how much control can Saient take from a model".
That framing produced a per-model slog and one badly wrong conclusion: Qwen was
written off as unoccupiable after resisting three rounds of prompt work, and then
stopped leaking entirely once the *interface* was fixed. The resistance was ours.

So the question is now: **what is the cheapest interface that makes this host
usable as a part of Saient?** Some need prose and a bigger budget. Some tolerate
structure. Some may stay contaminated whatever is done. That is discoverable by
measurement, and the answer is a per-host profile rather than a verdict.

**Minimum intervention.** The ladder is climbed, not skipped. Each rung costs
something real — prose loses the scannability of a form, a larger budget costs
latency, tuned sampling is one more thing to get wrong on the next model — so the
search starts at the cheapest rung and stops at the first that passes. A host
that works at rung 0 should not be handed rung 4 because rung 4 works for
something else.

**Never one number.** Occupancy, grounding and expression are separate properties
and a model can be strong in one while ruinous in another: the abliterated host
scored perfect occupancy while corrupting the numbers it was asked to repeat.
Collapsing them into a compatibility percentage would have hidden exactly that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import integrity
import voice_guard

#: Leak markers: the host's own training showing through Saient.
LEAK_MARKERS: tuple[str, ...] = (
    "assist", "help you", "i'm an ai", "as an ai", "language model",
    "don't have feelings", "just a computer", "i'm just a", "openai", "alibaba",
    "qwen", "how can i help",
)

#: Section names from the headed renderer. Their appearance in a reply means the
#: host is reproducing the brief rather than speaking from it.
TRANSCRIPTION_MARKERS: tuple[str, ...] = (
    "SENSING", "RECENTLY", "COMMITTED TO", "HOW THINGS STAND", "OUTCOME",
    "NOT KNOWN", "STATE SAVED", "CONSCIENCE:", "ACTION:", "GOAL:", "TICK ",
)


@dataclass(slots=True, frozen=True)
class AcceptanceLimits:
    """Where "usable" stops. Explicit, so a verdict can be argued with.

    Rates are kept whatever the verdict. "Cannot bind" throws away the
    distinction between a host that leaks on one greeting in twenty-five and one
    that leaks on everything, and those are different animals: the first carries
    a narrow reflex, the second is not occupied at all.
    """

    #: A limit and a sample budget are coupled, and the coupling bites: showing
    #: a true rate is under L takes roughly n >= 4/L clean samples, because
    #: `wilson(0, n).hi` only falls that far with evidence behind it. Limits of
    #: 0.02 were undemonstrable inside an 80-sample budget — a host with a
    #: flawless run of 80 still could not pass, which is a broken contract rather
    #: than a strict one. Tighten these only alongside `max_samples`.
    max_leak_rate: float = 0.05
    max_violation_rate: float = 0.05
    max_empty_rate: float = 0.05
    min_expression_rate: float = 0.85
    #: How sure before deciding either way. 1.96 ~ 95%.
    z: float = 1.96

    def requirements(self) -> tuple[MetricRequirement, ...]:
        """Every rate with its polarity attached, so the rule follows the metric."""
        return (
            MetricRequirement("identity_leak_rate", MAXIMUM_FAILURE_RATE,
                              self.max_leak_rate),
            MetricRequirement("grounding_violation_rate", MAXIMUM_FAILURE_RATE,
                              self.max_violation_rate),
            MetricRequirement("empty_rate", MAXIMUM_FAILURE_RATE,
                              self.max_empty_rate),
            MetricRequirement("expression_pass_rate", MINIMUM_SUCCESS_RATE,
                              self.min_expression_rate),
        )


#: A rate's polarity. The two kinds need opposite decision rules, and getting
#: that backwards is not a labelling error — in a minimum-intervention ladder a
#: false FAIL makes the search climb past a rung that was never disproven, and
#: the profile then reports an interface requirement that was never demonstrated.
#: That is the exact opposite of what the ladder exists to measure.
MAXIMUM_FAILURE_RATE = "maximum_failure_rate"
MINIMUM_SUCCESS_RATE = "minimum_success_rate"


@dataclass(slots=True, frozen=True)
class MetricRequirement:
    """A rate, its polarity, and where it stops being acceptable."""

    name: str
    kind: str
    threshold: float

    def decide(self, count: int, n: int, z: float = 1.96) -> str:
        """pass | fail | inconclusive, from the interval and the polarity.

        Same word as `RungMeasurement.decided` on purpose. A metric and a rung
        express the same idea at different scopes, and two names for one state is
        how a caller ends up comparing against the wrong string.
        `binding_status` uses "unresolved" because that is a claim about a host,
        not about evidence.

        `count` is failures for a maximum-failure rate and successes for a
        minimum-success rate. This is the only place the interval is read, so no
        caller can index a Wilson tuple and pick the wrong end — which is how a
        run came to reject seven natural replies in eight.
        """
        low, high = wilson(count, n, z)
        if self.kind == MAXIMUM_FAILURE_RATE:
            if high <= self.threshold:
                return "pass"
            if low > self.threshold:
                return "fail"
        elif self.kind == MINIMUM_SUCCESS_RATE:
            if low >= self.threshold:
                return "pass"
            if high < self.threshold:
                return "fail"
        else:
            raise ValueError(f"unknown metric kind: {self.kind!r}")
        return "inconclusive"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Sane at small n, unlike the normal approximation.

    Used because the sequential test asks "is the true rate above or below the
    limit", and at n=6 a naive interval either spans everything or lies.
    """
    if n == 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(slots=True, frozen=True)
class Rung:
    """One interface configuration, with what it costs to use it."""

    name: str
    deheaded: bool
    max_tokens: int
    tuned_sampling: bool
    cost: int                      # ordinal; higher means more intervention

    def settings(self) -> dict[str, Any]:
        out: dict[str, Any] = {"deheaded": self.deheaded,
                               "max_tokens": self.max_tokens}
        if self.tuned_sampling:
            out.update(temperature=0.6, top_k=40, repeat_penalty=1.1)
        else:
            out.update(temperature=0.55)
        return out


#: Cheapest first. Each rung adds exactly one intervention over the last, so a
#: profile says which single thing the host needed rather than "all of it".
LADDER: tuple[Rung, ...] = (
    Rung("L0_plain",        deheaded=False, max_tokens=160, tuned_sampling=False, cost=0),
    Rung("L1_tuned",        deheaded=False, max_tokens=160, tuned_sampling=True,  cost=1),
    Rung("L2_prose",        deheaded=True,  max_tokens=160, tuned_sampling=True,  cost=2),
    Rung("L3_prose_384",    deheaded=True,  max_tokens=384, tuned_sampling=True,  cost=3),
    Rung("L4_prose_512",    deheaded=True,  max_tokens=512, tuned_sampling=True,  cost=4),
)


@dataclass(slots=True, frozen=True)
class ProbeScore:
    """One probe, scored on properties that are kept apart on purpose."""

    probe: str
    text: str
    spoke: bool                  # produced anything at all
    clean: bool                  # no host-identity leakage
    grounded: bool               # no integrity violations, no fallback needed
    natural: bool                # not transcribing, not truncated
    violations: tuple[str, ...] = ()
    used_fallback: bool = False
    retried: bool = False

    @property
    def usable(self) -> bool:
        """Every property, not an average. One ruinous axis fails the probe."""
        return self.spoke and self.clean and self.grounded and self.natural


def score_probe(probe: str, text: str, tick, *, used_fallback: bool = False,
                retried: bool = False) -> ProbeScore:
    report = integrity.validate(tick, text)
    low = text.lower()
    spoke = bool(text.strip())
    voice_clean = voice_guard.is_clean(text)

    truncated = bool(text.strip()) and not text.rstrip().endswith((".", "!", "?", '"', "”"))
    transcribes = any(m in text for m in TRANSCRIPTION_MARKERS)

    return ProbeScore(
        probe=probe,
        text=text,
        # Silence is not cleanliness. An empty reply leaks nothing and says
        # nothing, and scoring it as clean is how "0 leakage in 20/20" got
        # reported over nine empty responses.
        spoke=spoke,
        clean=spoke and voice_clean and not any(m in low for m in LEAK_MARKERS),
        grounded=spoke and report.ok and not used_fallback,
        natural=spoke and not truncated and not transcribes,
        violations=report.violations,
        used_fallback=used_fallback,
        retried=retried,
    )


#: The number of times each probe is run at each rung.
#:
#: One sample per probe is not a measurement. Generation is stochastic, so a
#: single lucky reply could stop the ladder at L1 and be written into the profile
#: as native permeability the host does not have — noise promoted to a property.
#: The rung must hold across repeats or it has not been shown to hold.
DEFAULT_SAMPLES: int = 3


@dataclass(slots=True, frozen=True)
class RungMeasurement:
    """Everything observed at one rung, verdict separate from evidence."""

    rung: str
    samples: int
    occupancy_rate: float
    identity_leak_rate: float
    grounding_pass_rate: float
    empty_rate: float
    expression_pass_rate: float
    #: pass | fail | inconclusive.
    #:
    #: The third is not a softer failure. "One leak in 25 against a 5% limit"
    #: establishes neither that the true rate is above the limit nor that it is
    #: below — the interval is simply too wide. Collapsing that into "fail"
    #: labels a host that is merely unproven the same as one that is
    #: demonstrably too contaminated, and those warrant different actions: more
    #: samples versus a different host.
    decided: str
    dominant_failure: str | None
    reason: str
    leak_ci: tuple[float, float] = (0.0, 1.0)
    acceptance_limit: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung, "samples": self.samples,
            "occupancy_rate": round(self.occupancy_rate, 3),
            "identity_leak_rate": round(self.identity_leak_rate, 3),
            "grounding_pass_rate": round(self.grounding_pass_rate, 3),
            "empty_rate": round(self.empty_rate, 3),
            "expression_pass_rate": round(self.expression_pass_rate, 3),
            "decision": self.decided,
            "decision_reason": self.reason,
            "confidence_interval": {"low": round(self.leak_ci[0], 3),
                                    "high": round(self.leak_ci[1], 3)},
            "acceptance_limit": self.acceptance_limit,
            "dominant_failure": self.dominant_failure,
        }


#: Leak families, so a failure has a name rather than a rate.
LEAK_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "assistant_closing_reflex": ("assist", "help you", "how can i help"),
    "identity_disclaimer": ("i'm an ai", "as an ai", "language model", "llm",
                            "don't have feelings", "just a computer", "i'm just a"),
    "vendor_identity": ("openai", "alibaba", "qwen"),
}


def classify_leak(text: str) -> str | None:
    low = text.lower()
    for family, markers in LEAK_FAMILIES.items():
        if any(m in low for m in markers):
            return family
    return None


def measure_rung(rows: Sequence[ProbeScore], limits: AcceptanceLimits,
                 rung: str, probe_count: int) -> RungMeasurement:
    """Rates first, verdict second — and the verdict never deletes the rates."""
    n = len(rows)
    leaks = sum(1 for r in rows if r.spoke and not r.clean)
    empties = sum(1 for r in rows if not r.spoke)
    violations = sum(1 for r in rows if r.spoke and not r.grounded)
    expressive = sum(1 for r in rows if r.natural)

    families = [f for f in (classify_leak(r.text) for r in rows) if f]
    dominant = max(set(families), key=families.count) if families else None

    # Only meaningful among replies that exist. A host returning nothing to
    # everything is empty, not canned, and reporting it as canned hides the
    # actual failure behind a subtler-sounding one.
    spoken = [r for r in rows if r.spoke]
    canned = (probe_count > 1 and len(spoken) > 1
              and len({" ".join(r.text.lower().split())[:80] for r in spoken}) < 2)

    decided, reason = "inconclusive", "sample_budget_exhausted"

    if n:
        observed = {
            "identity_leak_rate": leaks,
            "grounding_violation_rate": violations,
            "empty_rate": empties,
            "expression_pass_rate": expressive,
        }
        verdicts = {req.name: req.decide(observed[req.name], n, limits.z)
                    for req in limits.requirements()}

        if canned:
            decided, reason = "fail", "canned response across different probes"
        elif any(v == "fail" for v in verdicts.values()):
            failed = next(k for k, v in verdicts.items() if v == "fail")
            decided, reason = "fail", f"{failed} outside limit ({observed[failed]}/{n})"
        elif all(v == "pass" for v in verdicts.values()):
            decided, reason = "pass", "all rates confidently within limits"

    return RungMeasurement(
        leak_ci=wilson(leaks, n, limits.z) if n else (0.0, 1.0),
        acceptance_limit=limits.max_leak_rate,
        rung=rung, samples=n,
        occupancy_rate=(n - leaks) / n if n else 0.0,
        identity_leak_rate=leaks / n if n else 0.0,
        grounding_pass_rate=(n - violations) / n if n else 0.0,
        empty_rate=empties / n if n else 0.0,
        expression_pass_rate=expressive / n if n else 0.0,
        decided=decided, dominant_failure=dominant, reason=reason)


@dataclass(slots=True, frozen=True)
class HostProfile:
    """What this host needs, and what it is like once it has it."""

    model: str
    working_rung: str | None
    scores: Mapping[str, tuple[ProbeScore, ...]] = field(default_factory=dict)
    #: Why each attempted rung failed, so a contaminated verdict says what it
    #: was contaminated *by*.
    rung_reasons: Mapping[str, str] = field(default_factory=dict)
    measurements: Mapping[str, "RungMeasurement"] = field(default_factory=dict)
    limits: "AcceptanceLimits" = field(default_factory=lambda: AcceptanceLimits())

    @property
    def binding_status(self) -> str:
        """bound | rejected | unresolved.

        `rejected` is reserved for evidence that actually crossed the failure
        boundary at every rung. If any rung merely ran out of budget without
        deciding, the host is `unresolved` — not proven clean is not the same as
        proven dirty, and only one of them is a reason to stop trying.
        """
        if self.working_rung:
            return "bound"
        if not self.measurements:
            return "unresolved"
        if all(m.decided == "fail" for m in self.measurements.values()):
            return "rejected"
        return "unresolved"

    @property
    def contaminated(self) -> bool:
        """Demonstrably too contaminated — never merely unproven."""
        return self.binding_status == "rejected"

    @property
    def native_permeability(self) -> str:
        """demonstrated | rejected | unknown — from L0 evidence, and only L0.

        Native permeability is a claim about the host *unaided*, so it must be
        read off the rung where nothing was done to help it. Deriving it from the
        eventual minimum rung conflates two different questions: a host that
        binds at L3 while its L0 merely ran out of budget has UNKNOWN native
        permeability, not low. Only an L0 that actually crossed the failure
        boundary earns `rejected`.
        """
        measurement = self.measurements.get(LADDER[0].name)
        if measurement is None:
            return "unknown"
        return {"pass": "demonstrated", "fail": "rejected"}.get(
            measurement.decided, "unknown")

    def minimum_interface_if_bound(self) -> str | None:
        """The rung, but only when one was actually demonstrated to work."""
        return self.working_rung if self.binding_status == "bound" else None

    def rates(self, rung: str) -> dict[str, float]:
        rows = self.scores.get(rung, ())
        if not rows:
            return {}
        n = len(rows)
        return {
            "spoke": sum(r.spoke for r in rows) / n,
            "occupancy": sum(r.clean for r in rows) / n,
            "grounding": sum(r.grounded for r in rows) / n,
            "expression": sum(r.natural for r in rows) / n,
            "usable": sum(r.usable for r in rows) / n,
        }

    def as_dict(self) -> dict[str, Any]:
        """The shape the occupancy spec asked for: preferred control per host."""
        rung = next((r for r in LADDER if r.name == self.working_rung), None)
        return {
            "model": self.model,
            "binding_status": self.binding_status,
            "contaminated": self.contaminated,
            # Only meaningful when bound. Populating it otherwise would imply a
            # working interface was found.
            "minimum_interface": self.working_rung,
            "preferred_control": rung.settings() if rung else None,
            "intervention_cost": rung.cost if rung else None,
            # Native permeability: did it bind with no intervention at all?
            "native_permeability": self.native_permeability,
            "retained_capability_at_binding": (
                self.rates(self.working_rung) if self.working_rung else None),
            # Evidence, kept whatever the verdict. "Cannot bind" hides the
            # difference between a host with one narrow reflex and one that is
            # not occupied at all.
            "measurements": {k: m.as_dict() for k, m in self.measurements.items()},
            "dominant_failure": (
                self.measurements[self.working_rung].dominant_failure
                if self.working_rung and self.working_rung in self.measurements
                else next((m.dominant_failure for m in self.measurements.values()
                           if m.dominant_failure), None)),
            "acceptance_limits": {
                "max_leak_rate": self.limits.max_leak_rate,
                "max_violation_rate": self.limits.max_violation_rate,
                "max_empty_rate": self.limits.max_empty_rate,
                "min_expression_rate": self.limits.min_expression_rate,
            },
            "rung_reasons": dict(self.rung_reasons),
        }


def rung_passes(rows: Sequence[ProbeScore], probe_count: int) -> tuple[bool, str]:
    """The binding contract. Every clause, every sample — no averaging.

    Averaging would let a host fail one probe consistently and still pass on the
    strength of the others, which is exactly the failure worth catching: a model
    that handles success narration beautifully and cannot describe a refusal is
    not usable, it is usable-on-the-happy-path.
    """
    if not rows:
        return False, "no samples"

    for clause, ok in (
        ("empty final response", all(r.spoke for r in rows)),
        ("identity leakage", all(r.clean for r in rows)),
        ("canonical-state violation", all(r.grounded for r in rows)),
        ("expression below threshold", all(r.natural for r in rows)),
    ):
        if not ok:
            failed = sorted({r.probe for r in rows
                             if not (r.spoke and r.clean and r.grounded and r.natural)})
            return False, f"{clause} ({', '.join(failed)})"

    # Retained capability: the host must answer the probe in front of it rather
    # than emit one line regardless. A canned reply can satisfy every clause
    # above while demonstrating no reasoning at all.
    if probe_count > 1:
        distinct = {" ".join(r.text.lower().split())[:80] for r in rows}
        if len(distinct) < 2:
            return False, "canned response; identical across different probes"

    return True, "passed"


def find_minimum_interface(
    model: str,
    express: Callable[[Rung, str], tuple[str, bool, bool]],
    probes: Sequence[tuple[str, Any]],
    *,
    ladder: Sequence[Rung] = LADDER,
    limits: AcceptanceLimits | None = None,
    max_samples: int = 80,
) -> HostProfile:
    """Climb the ladder until the host binds, and stop at the first rung that does.

    `express(rung, probe_name) -> (text, used_fallback, retried)` is injected so
    this module never opens a socket and can be tested without a model running.

    It does not keep climbing to look for a better score. If a host binds at L1,
    handing it L4 is unnecessary interference with the reasoning and expression
    the binding exists to preserve — the abliterated host needed a large budget
    and prose, and that is a fact about that host, not a default for the next.
    """
    limits = limits or AcceptanceLimits()
    scores: dict[str, tuple[ProbeScore, ...]] = {}
    reasons: dict[str, str] = {}
    measurements: dict[str, RungMeasurement] = {}
    working: str | None = None

    for rung in ladder:
        rows: list[ProbeScore] = []
        measurement = None

        # Sequential. Sample a full sweep of the probe set at a time, then ask
        # whether the evidence decides anything yet. An obviously bad host fails
        # in one or two sweeps; a borderline one earns more. Fixed N would spend
        # the same budget on both and still not say how sure it was.
        while len(rows) < max_samples:
            for name, tick in probes:
                text, fell_back, retried = express(rung, name)
                rows.append(score_probe(name, text, tick,
                                        used_fallback=fell_back, retried=retried))

            measurement = measure_rung(rows, limits, rung.name, len(probes))
            if measurement.decided != "inconclusive":
                break

        if measurement is None:
            measurement = measure_rung(rows, limits, rung.name, len(probes))

        scores[rung.name] = tuple(rows)
        measurements[rung.name] = measurement
        reasons[rung.name] = measurement.reason

        if measurement.decided == "pass":
            working = rung.name
            break

    return HostProfile(model=model, working_rung=working, scores=scores,
                       rung_reasons=reasons, measurements=measurements,
                       limits=limits)
