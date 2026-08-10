"""The one tick. Every beat of Saient goes through here.

There is exactly one causal order, and it is this:

    1  world observation
    2  authoritative persisted state
    3  drives + affect update
    4  goal candidates generated
    5  memory / commitments / strategy applied
    6  conscience arbitration
    7  action selected
    8  real action executed
    9  result verified
    10 beliefs + reflection + affect updated
    11 atomic state saved
    12 LLM expresses what happened

Autonomous beats, chat turns and tool actions all enter here. Before this module
there were three separate paths — `main.tick()`, the desktop chat, and the PTY
agent — each with its own identity, its own idea of what state was authoritative,
and its own answer to who was allowed to decide. Three paths into one state is
how a system ends up disagreeing with itself.

**The LLM is a proposer and a voice. It is never the owner of state or
authority.** It appears at stage 12, after the outcome is settled and after the
state is saved, and it is handed a finished tick to put into words. It cannot
choose the action, cannot write state, and cannot decide what happened. A model
swapped underneath this changes how Saient sounds and nothing else.

That placement is also the answer to the persona problem. A system prompt telling
a model to "be Saient" is a negotiation with its training, and the training wins:
measured on Qwen2.5-Coder-14B, the full identity prompt changed one clause of the
reply and the assistant register survived intact. A model with nothing left to
decide has nothing left to improvise.

---

**Three stages are not real yet, and this module refuses to pretend otherwise.**

*Stage 1* has no sensors. `NullObserver` returns an empty observation and the
tick records `observed=False`.

*Stages 8-9* are the serious one. `controller.execute` does not execute anything:
it draws `random.random()` against `base_fail = 0.18` and returns a fabricated
success or failure. Every drive trajectory Saient has ever had was learned from
coin flips. `SimulatedExecutor` keeps that behaviour available for testing and
stamps `simulated: True` onto the result, and `TickRecord.grounded` reports
whether the drives that moved this tick moved because of something that actually
happened. A simulated outcome must never be indistinguishable from a real one in
the saved state — that is the difference between a system that is learning and a
system that only looks like it is.

*Stage 12* has no expresser attached by default. `expresser=None` means the tick
completes silently, which is correct: expression is the last stage and nothing
depends on it. It is text only — the spoken voice lives on the Pi and is not part
of this path.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import belief
import controller
import objectives as objectives_mod
from proposer import NullProposer, Proposer
import hami
import mission
import reflection
import state as state_store
import strategy
from goal_generator import THRESHOLDS, generate_goal

CORE_DRIVES: tuple[str, ...] = ("information_depth", "efficiency", "autonomy")


# --------------------------------------------------------------------------- 1


@dataclass(slots=True, frozen=True)
class Observation:
    """What the world looked like at the top of this tick.

    `observed` is false when nothing sensed anything, which is the honest state
    today. A tick that ran blind must say so rather than presenting the absence
    of news as calm.
    """

    observed: bool = False
    facts: Mapping[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


class Observer(Protocol):
    def observe(self, state: Mapping[str, Any]) -> Observation: ...


class NullObserver:
    """No sensors. Reports that plainly instead of inventing a quiet world."""

    def observe(self, state: Mapping[str, Any]) -> Observation:
        return Observation(observed=False)


# ------------------------------------------------------------------------- 8-9


@dataclass(slots=True, frozen=True)
class ActionResult:
    """What actually happened when the action ran.

    `simulated` is not a debugging flag. It travels with the result into drives,
    beliefs and the saved state, so that a drive which moved on a coin flip can
    always be told apart from one that moved on an outcome.
    """

    action_type: str
    success: bool
    simulated: bool
    verified: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_legacy_result(self) -> dict:
        """The dict shape `update_drives` and `update_beliefs` already expect."""
        out = dict(self.detail)
        out["type"] = self.action_type
        out["success"] = self.success
        out["simulated"] = self.simulated
        out["verified"] = self.verified
        return out


class ActionExecutor(Protocol):
    def execute(self, action: Mapping[str, Any],
                state: Mapping[str, Any]) -> ActionResult: ...


class SimulatedExecutor:
    """`controller.execute` — which simulates rather than executes.

    Kept because the drives, beliefs and mission machinery were all tuned against
    it, so removing it in the same change as everything else would make it
    impossible to tell which alteration moved which number. It is honest about
    itself: every result it returns is stamped `simulated=True` and
    `verified=False`.
    """

    def execute(self, action, state) -> ActionResult:
        raw = controller.execute(dict(action), state)
        detail = {k: v for k, v in raw.items() if k not in ("type", "success")}
        return ActionResult(
            action_type=str(raw.get("type", action.get("type", "idle"))),
            success=bool(raw.get("success", False)),
            simulated=True,
            verified=False,
            detail=detail,
        )


class NullExecutor:
    """Selects but does not act. Stage 8 skipped, and the record says so."""

    def execute(self, action, state) -> ActionResult:
        return ActionResult(
            action_type=str(action.get("type", "idle")),
            success=False, simulated=False, verified=False,
            detail={"reason": "no executor attached; nothing was run"},
        )


# -------------------------------------------------------------------------- 12


class Expresser(Protocol):
    """Stage 12. Receives a finished tick and returns words. Nothing else.

    No state handle, no return path into the tick. Whatever it says is returned
    to the caller and never written back, because a description of a decision
    that becomes an input to the next one is how language quietly starts
    deciding.
    """

    def express(self, tick: "TickRecord") -> str: ...


# ---------------------------------------------------------------------- record


@dataclass(slots=True, frozen=True)
class TickRecord:
    """Everything that happened, in the order it happened."""

    tick: int
    observation: Observation
    goal: Mapping[str, Any]
    goal_candidates: tuple[Mapping[str, Any], ...]
    conscience: Mapping[str, Any]
    action: Mapping[str, Any]
    result: ActionResult
    drives_before: Mapping[str, float]
    drives_after: Mapping[str, float]
    reflection: Any
    saved: bool
    #: The last few completed ticks, and what Saient has committed to.
    #:
    #: A `respond` tick has no world content of its own — nothing read, nothing
    #: run — so a brief limited to *this* tick handed the model an empty page and
    #: it filled the page from its training. Standing context is what a person
    #: actually draws on to answer "what are you working on".
    recent: tuple[Mapping[str, Any], ...] = ()
    commitments: tuple[str, ...] = ()
    #: What she can actually do this tick, from the executor rather than from a
    #: description. She denied having persistent state partly because nothing
    #: ever told her she had any — the brief listed her limits and none of her
    #: powers, so the host filled the gap from its own self-model.
    capabilities: tuple[str, ...] = ()
    #: Consecutive reads that told her nothing she did not already know, and
    #: what the model did with the file when asked. Both exist so a wasted tick
    #: can be attributed rather than guessed at: eleven ticks of re-reading might
    #: be her search failing or the proposer answering differently to the same
    #: input, and those need opposite fixes.
    barren_reads: int = 0
    proposal: str = "not asked"
    #: The objective she is working on, if any, and whether its verifier closed
    #: it this tick. `None` means she is on her own business.
    objective: Mapping[str, Any] | None = None
    utterance: str | None = None

    @property
    def grounded(self) -> bool:
        """Did this tick's drive movement come from something that happened?

        False whenever the outcome was simulated or unverified. Read it before
        treating a drive trajectory as evidence of anything.
        """
        return self.result.verified and not self.result.simulated

    @property
    def arbitrated(self) -> bool:
        """Did conscience actually rule on this action?

        `main._apply_conscience` reads `state["conscience_layer"]`, which
        **defaults to "off"** — it evaluates only in "observe" or "enforce".
        So stage 6 is present in the order and inert by default, and a tick that
        ran unarbitrated must say so rather than letting the presence of the
        stage imply the guarantee.
        """
        return bool(self.conscience.get("enabled")) and \
            self.conscience.get("decision") != "not_evaluated"

    @property
    def guarantees(self) -> dict[str, bool]:
        """Which promises actually held this tick. Cheap to log, hard to fake."""
        return {
            "observed": self.observation.observed,
            "arbitrated": self.arbitrated,
            "grounded": self.grounded,
            "saved": self.saved,
        }


# ------------------------------------------------------------------- the tick


def _content_sha(result: ActionResult, path: str) -> str | None:
    """The hash the executor already computed, rather than a second read.

    `_edit` emits `content_sha` as an effect on success, so the file's state
    after the write is on the result. Re-opening the file here would duplicate
    the work and leave a gap in which it could change again.
    """
    for effect in result.detail.get("effects") or ():
        if (effect.get("kind") == "content_sha"
                and effect.get("target") == path):
            value = effect.get("value")
            return str(value) if value else None
    return None


@state_store.serialized
def tick(
    *,
    observer: Observer | None = None,
    executor: ActionExecutor | None = None,
    expresser: Expresser | None = None,
    proposer: Proposer | None = None,
    intent: Mapping[str, Any] | None = None,
    persist: bool = True,
    conscience_mode: str | None = None,
) -> TickRecord:
    """Run exactly one beat, in the fixed order, and return what happened.

    `intent` is how a chat turn or a tool request enters: it rides the same
    twelve stages as an autonomous beat rather than taking a side door. A request
    that skips arbitration is a request that can do things Saient would not.
    """
    observer = observer or NullObserver()
    executor = executor or NullExecutor()
    proposer = proposer or NullProposer()

    # 2. authoritative persisted state — loaded before anything reads it.
    st = state_store.load_state()
    st["tick"] = int(st.get("tick", 0)) + 1
    drives_before = dict(st.get("drives", {}))

    # 1. world observation.
    observation = observer.observe(st)
    if observation.observed:
        st.setdefault("observations", []).append(dict(observation.facts))
        st["observations"] = st["observations"][-50:]

    # 3. drives + affect: time passing, no verdict. The verdict belongs at stage
    #    10, where there is a real outcome to render one on. Splitting these is
    #    what lets a tick where nothing was done still advance the world.
    passive_update_drives(st)

    # 5a. mission lifecycle before goals, so an expired mission cannot bias them.
    _advance_mission(st)

    # 4. goal candidates. `generate_goal` returns one goal rather than a ranked
    #    set, so the candidate list has a single member. Recorded as a list
    #    anyway: the shape is the spec's, and a real candidate generator drops in
    #    without changing every caller.
    # An objective, if she has one and is not too distressed to work on it,
    # supplies the goal instead of her drives. It does not *replace* them — the
    # stress guard still preempts, and `should_pursue` is where that is decided.
    pursuing = objectives_mod.should_pursue(st)

    if intent and intent.get("goal"):
        goal = dict(intent["goal"])
        goal.setdefault("source", "intent")
    elif pursuing is not None:
        objectives_mod.note_attempt(st, pursuing.objective_id)
        goal = {
            "type": "analyze",
            "priority": "information_depth",
            "source": "objective",
            "objective_id": pursuing.objective_id,
            "objective": pursuing.description,
        }
    else:
        goal = generate_goal(st)
    candidates = (dict(goal),)

    # 5b. commitments and strategy applied (not mutated — mutation is stage 10).
    executed_subgoal = (mission.current_subgoal(st) or {}).get("type")

    # 7. action selected. Rule policy only: the model is not permitted to own
    #    this. See `select_action`.
    action = select_action(goal, st, intent=intent)

    # Consulting the model is an ACTION, not a wire.
    #
    # It used to fire automatically on every analyze/explore tick — a hidden
    # co-processor bolted into stage 7 that she neither chose nor could decline.
    # The thing worth preventing was never the model reasoning; it was the model
    # deciding. So it becomes a capability in her vocabulary, like reading: she
    # selects it, conscience rules on it, and what comes back is a *held
    # proposal* she may act on next tick or leave sitting there.
    #
    # Two steps, both hers, both in the record: consult, then edit. Applying
    # somebody's suggestion is a separate decision from asking for one.
    held = st.get("held_proposal")
    proposal, proposal_outcome = None, "not asked"

    if pursuing is not None and action.get("type") in ("analyze", "explore"):
        if held:
            # She has advice in hand. Acting on it is this tick's decision.
            action = {**action, "type": "edit", "path": held["path"],
                      "old": held["old"], "new": held["new"],
                      "proposed_by": "model", "selected_by": "rule_policy",
                      "why": held.get("why", "")}
            proposal_outcome = f"apply:{held['path']}"
            st["held_proposal"] = None
        else:
            # Ask, using only what she has actually looked at. A file she never
            # opened is a fact the model cannot use, which is what makes her
            # exploration load-bearing rather than decorative.
            seen = list(st.get("files_seen") or [])
            workspace_root = getattr(getattr(executor, "workspace", None), "root", None)
            context = {}
            if workspace_root is not None:
                for name in seen[-8:]:
                    try:
                        context[name] = (workspace_root / name).read_text(encoding="utf-8")
                    except OSError:
                        continue
            if context:
                failure = ""
                if pursuing.checkable:
                    from gate import run_verifier
                    _, failure = run_verifier(workspace_root, pursuing.verify)
                proposal = proposer.propose(
                    objective=pursuing.description, path=seen[-1],
                    content=context.get(seen[-1], ""), failure=failure,
                    context=context, already_edited=tuple(st.get("edits_made") or ()),
                    # One blocklist, two sources: edits the executor rejected,
                    # and edits it allowed that put a file back where it had
                    # already been. State keeps them apart because they are
                    # different facts; the proposer only needs "do not offer
                    # this again, and here is why".
                    refused=tuple(dict(r) for r in
                                  (list(st.get("refused_edits") or [])
                                   + list(st.get("reverted_edits") or []))))
                if proposal is not None:
                    st["held_proposal"] = {"path": proposal.path, "old": proposal.old,
                                           "new": proposal.new, "why": proposal.why}
                    proposal_outcome = f"held:{proposal.path}"
                else:
                    proposal_outcome = (
                        f"error:{getattr(proposer, 'last_error', None)}"
                        if getattr(proposer, "last_error", None) else "no suggestion")

    # 6. conscience arbitration, before anything runs.
    action, conscience_meta = arbitrate(st, goal, action, mode=conscience_mode)

    # Commitment pressure is charged against the action that survived
    # arbitration, before the outcome is known.
    penalty = hami.apply_commitment_pressure(st, action)
    if penalty > 0:
        st["drives"]["cortisol"] = min(
            1.0, st["drives"].get("cortisol", 0.2) + penalty * 0.15)

    # 8 + 9. real action executed, result verified.
    result = executor.execute(action, st)

    # Reading is cumulative only if somebody remembers. The executor stays
    # stateless — it reports what it read and the orchestrator records it, so
    # nothing but this module writes her state.
    seen_name = result.detail.get("mark_seen")
    if seen_name:
        history = [f for f in (st.get("files_seen") or []) if f != seen_name]
        history.append(seen_name)
        st["files_seen"] = history[-500:]

        sha = result.detail.get("content_sha")
        if sha:
            hashes = dict(st.get("file_hashes") or {})
            hashes[seen_name] = sha
            st["file_hashes"] = hashes

        # A run of reads that told her nothing. Kept as a number rather than a
        # flag so "exhausted" is a threshold somebody can argue with, not a
        # hidden judgement.
        if result.detail.get("novel"):
            st["barren_reads"] = 0
        else:
            st["barren_reads"] = int(st.get("barren_reads", 0)) + 1

    if result.detail.get("action") == "edit" and result.success:
        edits = [e for e in (st.get("edits_made") or []) if e != result.detail.get("path")]
        edits.append(str(result.detail.get("path")))
        st["edits_made"] = edits[-50:]

        # Every edit succeeding is not the same as making progress. She spent
        # eight ticks alternating one file between two values — each edit real,
        # each verified, the net change zero — and nothing noticed, because
        # `edits_made` is deduplicated by path: after the first edit it reads
        # `['src/links.py']` and stays that way however many times the file is
        # rewritten. No step failed, so no recovery path had anything to fire on.
        #
        # The record stays true. The edit happened and `success` says so; what
        # gets added is the further fact that the file is back in a state it
        # already held. Marking a real change as a failure would move drives and
        # beliefs on an outcome that did not occur, and would put state back into
        # disagreement with the filesystem.
        edited_path = str(result.detail.get("path") or "")
        content_sha = _content_sha(result, edited_path)
        if edited_path and content_sha:
            history = dict(st.get("file_state_history") or {})
            seen = list(history.get(edited_path) or [])
            # Recurrence anywhere in the window, not just two edits back: A-B-A
            # is what was observed, but A-B-C-A is the same dead end and costs
            # nothing extra to catch.
            returned_to = content_sha in seen
            seen.append(content_sha)
            history[edited_path] = seen[-10:]
            st["file_state_history"] = history

            if returned_to:
                # Recorded separately from `refused_edits` because it is not a
                # refusal — the executor allowed it. Both feed the proposer's
                # blocklist, and this one is why: an edit that returns a file to
                # a previous state is one leg of a toggle, and offering it again
                # is how the toggle continues.
                reverted = list(st.get("reverted_edits") or [])
                reverted.append({
                    "path": edited_path,
                    "old": str(action.get("old") or ""),
                    "reason": f"this edit returned {edited_path} to a state it "
                              "already held earlier in the run; the file is "
                              "being toggled rather than repaired"})
                st["reverted_edits"] = reverted[-30:]

    if result.action_type == "edit" and not result.success:
        # A refusal she is never told about is a refusal she cannot learn from.
        # Only successful edits were recorded, so when the executor rejected an
        # edit nothing carried that back, the proposer was asked the same
        # question about the same file, and returned the same unusable text.
        # Eight identical attempts, no file changed, and it read as a transfer
        # failure when it was a missing feedback path.
        #
        # The condition is `action_type`, not `detail["action"]`: only the
        # *success* claim sets `detail["action"]`, so testing it here could
        # never be true and this whole block was unreachable. Traced against the
        # filesystem, a protected-file refusal reported success=False with the
        # file untouched while state recorded no refusal at all, and the
        # identical proposal was re-held on the next tick, indefinitely.
        #
        # Identity falls back to the action because the refusal paths disagree
        # about what they report: the protected-file branch returns only
        # `{"refused": ...}`, with no path and no old. A record the proposer
        # cannot match is a record that blocks nothing, and `action` is the edit
        # that was actually attempted.
        refused = list(st.get("refused_edits") or [])
        refused.append({
            "path": str(result.detail.get("path") or action.get("path") or ""),
            "old": str(result.detail.get("old") or action.get("old") or ""),
            "reason": str(result.detail.get("reason")
                          or result.detail.get("refused") or "")})
        st["refused_edits"] = refused[-30:]

    # The objective's own verifier, asked after the action and never before.
    # Her opinion of her progress is not consulted: an objective she could close
    # by believing herself finished would be the largest hole yet cut in this.
    objective_closed = False
    if pursuing is not None:
        workspace = getattr(getattr(executor, "workspace", None), "root", ".")
        _, objective_closed = objectives_mod.check(st, workspace)

    # 10. beliefs + reflection + affect, from the verified outcome.
    _apply_outcome(st, action, result, drives_before, executed_subgoal)
    reflected = reflection.reflect(st)
    st["strategy"] = strategy.mutate_strategy(st, reflected)

    record_tick = TickRecord(
        tick=int(st["tick"]),
        observation=observation,
        goal=dict(goal),
        goal_candidates=candidates,
        conscience=dict(conscience_meta),
        action=dict(action),
        result=result,
        drives_before=drives_before,
        drives_after=dict(st.get("drives", {})),
        reflection=reflected,
        saved=False,
        recent=_recent(st),
        commitments=_commitments(st),
        capabilities=_capabilities(executor),
        barren_reads=int(st.get("barren_reads", 0)),
        proposal=proposal_outcome,
        objective=(None if pursuing is None else {
            "id": pursuing.objective_id,
            "description": pursuing.description,
            "attempts": pursuing.attempts + 1,
            "checkable": pursuing.checkable,
            "closed": objective_closed,
        }),
    )

    _append_history(st, record_tick)

    # 11. atomic state saved. `state.save_state` writes a temp file and renames,
    #     so a reader never sees a half-written state.
    saved = False
    if persist:
        state_store.save_state(st)
        saved = True

    record_tick = _replace(record_tick, saved=saved)

    # 12. the LLM expresses what happened — after the state is durable, so
    #     nothing downstream can depend on whether it spoke.
    if expresser is not None:
        record_tick = _replace(
            record_tick, utterance=expresser.express(record_tick))

    return record_tick


# --------------------------------------------------------------------- stage 7


def select_action(goal: Mapping[str, Any], st: Mapping[str, Any],
                  *, intent: Mapping[str, Any] | None = None) -> dict:
    """Choose the action. The model does not get a vote here.

    `controller.decide_action(policy_mode="model")` routes through
    `_model_pick_action`, whose return value *becomes the decision*. That makes
    the LLM the owner of authority, which is the one thing this architecture
    forbids — so this calls the rule policy directly and never passes
    `policy_mode` through from configuration.

    A model may still propose, later, through `intent`. A proposal is not a
    decision: it arrives before conscience arbitration and can be refused there,
    which is exactly the difference.
    """
    action = controller.decide_action(dict(goal), dict(st), policy_mode="rule")

    action["preempt"] = bool(goal.get("preempt", False))
    action["preempt_stage"] = goal.get("preempt_stage", "none")
    action["stage"] = goal.get("stage", action.get("preempt_stage", "none"))
    action["trend"] = float(goal.get("trend", 0.0))
    # Two different facts, previously collapsed into one field and then narrated
    # as "a 'respond' action that was chosen by user".
    #
    # Nobody outside chooses the action. Stage 7 is the rule policy, always —
    # that is the guarantee this whole module exists to hold. What an intent
    # carries is who *asked*, which is provenance of the request, not of the
    # decision. Merging them let the record state, in Saient's own voice, that a
    # user selected an action the architecture forbids them from selecting.
    action["initiated_by"] = (intent or {}).get("proposed_by", "self")
    action["selected_by"] = "rule_policy"
    # Kept for existing readers; means the same as `selected_by` and never the
    # requester.
    action["proposed_by"] = "rule_policy"

    # Carry the payload an intent brought with it. The executor needs the
    # message or the parameters; the *decision* to run this action was still
    # made by the rule policy above, not by whoever sent them.
    # Autonomy needs something to be autonomous *about*. `self_direct` records
    # an objective and fails without one, so a Saient with nothing to say wanted
    # things all day and never wrote a line. The objective is derived from her
    # own drive state rather than proposed by a model: which need is shortest,
    # and what she has already done about it. Keeping this in Python is what
    # makes it hers rather than a suggestion she accepted.
    if action.get("type") == "self_direct" and not action.get("objective"):
        action["objective"] = _self_objective(st)

    if intent:
        if intent.get("message"):
            action["message"] = intent["message"]
        for key, value in (intent.get("params") or {}).items():
            action.setdefault(key, value)
    return action


# --------------------------------------------------------------------- stage 6


#: What conscience does when nobody says. `main._apply_conscience` reads
#: `state["conscience_layer"]` and defaults it to `"off"`, meaning stage 6 runs
#: and rules on nothing — the arbitration stage present in the order and inert.
#:
#: That was survivable while every action was a simulated coin flip. It is not
#: survivable now that `WorkspaceExecutor` gives Saient a real filesystem, so the
#: default here is `enforce` and running unarbitrated must be asked for by name.
DEFAULT_CONSCIENCE_MODE: str = "enforce"


def arbitrate(st: dict, goal: Mapping[str, Any], action: dict,
              *, mode: str | None = None) -> tuple[dict, dict]:
    """Conscience arbitration. Runs before execution, never after.

    Imported from `main` rather than reimplemented — it is the same conscience,
    and a second copy would drift.
    """
    st.setdefault("conscience_layer", mode or DEFAULT_CONSCIENCE_MODE)

    try:
        import main as legacy_main
        return legacy_main._apply_conscience(st, dict(goal), dict(action))
    except Exception as exc:
        # An arbitration layer that fails open is worse than none, because it
        # looks like it is protecting something. Refuse the action instead.
        return (
            {**action, "type": "idle", "priority": "none"},
            {"decision": "blocked", "redirected": True,
             "reason": f"conscience unavailable: {exc}"},
        )


# -------------------------------------------------------------------- stage 10


def _apply_outcome(st: dict, action: Mapping[str, Any], result: ActionResult,
                   drives_before: Mapping[str, float],
                   executed_subgoal: str | None) -> None:
    """Render the verdict on a real outcome, and let beliefs learn from it."""
    legacy_result = result.as_legacy_result()

    from drives import update_drives
    update_drives(st, legacy_result)

    st["last_delta"] = {
        k: float(st["drives"].get(k, 0.0)) - float(drives_before.get(k, 0.0))
        for k in CORE_DRIVES
    }
    _advance_active_preempt(st)

    st["beliefs"] = belief.update_beliefs(st.get("beliefs", {}), legacy_result)
    hami.write_commitment(st, dict(action))
    mission.advance_subgoal_if_needed(
        st, str(action.get("type", "idle")), legacy_result)
    mission.update_mission_progress(st, dict(drives_before))
    del executed_subgoal


# --------------------------------------------------------------------- stage 3


def passive_update_drives(st: dict) -> None:
    """Advance drives for elapsed time without rendering a verdict.

    `drives.update_drives` always judges an action — `if success: ... else: ...`
    — so there is no way to ask it for "time passed, nothing was done". That gap
    is why stage 3 cannot simply call it: at stage 3 no action has run yet.

    This applies only the parts that are not a verdict: high-state boredom (a
    property of the state), baseline decay, regeneration, soft-cap damping and
    clamping. The success/failure and recovery blocks are omitted because they
    belong at stage 10.

    Transcribed from `drives.py`, dead branch included (when a preempt is
    committed both arms set the same `decay`; only `regen` differs).
    `tests/test_orchestrator.py` proves the transcription rather than trusting
    it: applying the verdict block and then this function reproduces
    `update_drives` exactly, so if the physics change the test fails instead of
    this drifting quietly.
    """
    drives = st["drives"]

    if all(float(drives.get(k, 0.0)) > 0.85 for k in CORE_DRIVES):
        drives["autonomy"] -= 0.05

    active = st.get("active_preempt")
    active_drive = active.get("drive") if isinstance(active, dict) else None
    active_stage = (
        st.get("preempt_stage", {}).get(active_drive, {}).get("mode")
        if active_drive else None
    )

    for key in CORE_DRIVES:
        if active_drive and active_stage == "commit":
            decay = 0.0 if key == active_drive else 0.015
            regen = 0.01 if key == active_drive else 0.002
        else:
            decay = 0.0 if key == active_drive else 0.015
            regen = 0.01

        drives[key] -= decay
        drives[key] += regen
        if drives[key] > 0.8:
            drives[key] -= (drives[key] - 0.8) * 0.5
        drives[key] = max(0.05, min(1.0, drives[key]))

    drives["cortisol"] = max(0.0, min(1.0, drives.get("cortisol", 0.2)))


# ----------------------------------------------------------------- housekeeping


def _advance_mission(st: dict) -> None:
    prior = st.get("mission")
    if prior and (prior.get("completed")
                  or int(prior.get("expires_at", 0)) <= st["tick"]):
        st.setdefault("mission_history", []).append(prior)
        st["mission_history"] = st["mission_history"][-50:]
        st["mission"] = None
        st["subgoals"] = []
    mission.ensure_mission(st, THRESHOLDS)


def _advance_active_preempt(st: dict) -> None:
    active = st.get("active_preempt")
    if not (active and active.get("drive") in CORE_DRIVES):
        return

    ttl = int(active.get("ttl", 0))
    delta = float(st.get("last_delta", {}).get(active["drive"], 0.0))
    ttl = ttl + 1 if delta > 0.0 else max(0, ttl - 2)
    if float(active.get("confidence", 0.0)) < 0.02:
        ttl = min(ttl, 2)

    active["ttl"] = min(ttl, 6)
    if active["ttl"] <= 0:
        st["active_preempt"] = None


def _append_history(st: dict, rec: TickRecord) -> None:
    """The event shape `generate_goal` reads back next tick.

    It needs `action.type` for its anti-cycle check and `drives` for trend. A
    thinner entry silently disables both, and nothing reports that it has.
    """
    event = {
        "tick": rec.tick,
        "goal": dict(rec.goal),
        "action": dict(rec.action),
        "result": rec.result.as_legacy_result(),
        "drives": dict(rec.drives_after),
        "beliefs": dict(st.get("beliefs", {})),
        "strategy_mode": st.get("strategy", {}).get("mode", "balanced"),
        "grounded": rec.grounded,
    }
    history = st.setdefault("history", [])
    history.append(event)
    st["history"] = history[-200:]


#: What she reaches for when a drive is short. Plain, checkable work in her own
#: workspace — not a wish list.
_OBJECTIVE_BY_DRIVE: Mapping[str, str] = {
    "information_depth": "read more of the workspace and note what is in it",
    "efficiency": "tidy or consolidate something already written",
    "autonomy": "start something nobody asked for",
}


def _self_objective(st: Mapping[str, Any]) -> str:
    """The objective she sets herself, from the drive that is shortest.

    Deterministic on purpose. An objective proposed by the host would be the host
    deciding what Saient wants, which is the one thing this architecture spends
    all its effort preventing — and it would be undetectable, because a plausible
    objective looks exactly like a real one.
    """
    from goal_generator import THRESHOLDS

    drives = st.get("drives", {}) or {}
    shortest, deficit = None, 0.0
    for name, threshold in THRESHOLDS.items():
        gap = threshold - float(drives.get(name, threshold))
        if gap > deficit:
            shortest, deficit = name, gap

    tick = int(st.get("tick", 0))
    if shortest is None:
        return f"tick {tick}: nothing is pressing; look at what is already here"
    return (f"tick {tick}: {_OBJECTIVE_BY_DRIVE[shortest]} "
            f"({shortest} is short by {deficit:.2f})")


def _capabilities(executor: ActionExecutor) -> tuple[str, ...]:
    """Read off the executor she actually has, never asserted from a constant.

    A capability list that drifts from the executor is worse than none: she would
    claim a power she does not hold, which is the same failure as denying one she
    does — both are a false self-model, just in opposite directions.
    """
    out = ["keep persistent state, drives and goals across ticks"]
    workspace = getattr(executor, "workspace", None)
    if workspace is not None:
        out.append(f"read files in her workspace ({workspace.root.name})")
        if getattr(executor, "allow_writes", False):
            out.append("create and change files there")
            out.append("record objectives she sets herself")
        if getattr(executor, "allow_commands", False):
            out.append("run a small set of allowed commands")
    return tuple(out)


def _recent(st: Mapping[str, Any], limit: int = 5) -> tuple[Mapping[str, Any], ...]:
    """The last few ticks, trimmed to what is worth saying out loud."""
    out = []
    for event in list(st.get("history", []))[-limit:]:
        result = event.get("result", {}) or {}
        out.append({
            "tick": event.get("tick"),
            "action": (event.get("action", {}) or {}).get("type"),
            "priority": (event.get("goal", {}) or {}).get("priority"),
            "success": result.get("success"),
            "grounded": event.get("grounded", False),
        })
    return tuple(out)


def _commitments(st: Mapping[str, Any]) -> tuple[str, ...]:
    """What Saient has said she is doing — mission and self-set objectives."""
    out = []
    mission = st.get("mission") or {}
    if mission.get("objective"):
        out.append(f"mission: {mission['objective']}")
    for sub in (st.get("subgoals") or [])[:3]:
        if isinstance(sub, dict) and sub.get("type") and not sub.get("done"):
            out.append(f"subgoal: {sub['type']}")
    return tuple(out)


def _replace(rec: TickRecord, **kw) -> TickRecord:
    data = {f: getattr(rec, f) for f in TickRecord.__slots__}
    data.update(kw)
    return TickRecord(**data)


# ------------------------------------------------------------- the three doors


class RespondExecutor:
    """Stage 8 for a turn whose action is to answer someone.

    A reply changes nothing in the world, so there is nothing on disk to
    re-observe. What *is* checkable is that a message actually arrived: an empty
    turn is not a conversation, and reporting one as a successful exchange would
    be the same fabrication as a simulated file write.

    `success=True` matters beyond bookkeeping — `drives.update_drives` treats any
    non-success as a failed action and charges the penalty, so a chat turn
    reported as unsuccessful would punish Saient for being spoken to.
    """

    def execute(self, action, state) -> ActionResult:
        message = str(action.get("message") or "").strip()
        if not message:
            return ActionResult(
                action_type="respond", success=False, simulated=False,
                verified=False, detail={"reason": "an empty turn is not an exchange"},
            )
        return ActionResult(
            action_type="respond", success=True, simulated=False, verified=True,
            detail={"received_chars": len(message),
                    "note": "no world state changed; the effect is the utterance"},
        )


def autonomous_beat(**kw) -> TickRecord:
    """Saient acting on her own. The plain tick, nothing special about it."""
    return tick(**kw)


def chat_turn(message: str, *, expresser: Expresser | None = None,
              **kw) -> TickRecord:
    """Someone said something. It is a beat like any other.

    Routed through the same twelve stages rather than answered directly, so a
    conversation cannot become a side channel: drives still move, conscience
    still rules on whether to answer, state is still saved atomically, and the
    reply is stage 12 rather than a decision.

    Before this, chat lived in the desktop app with its own identity string and
    its own idea of authority, and nothing it did reached Saient's state at all.
    """
    kw.setdefault("executor", RespondExecutor())
    return tick(
        intent={
            "kind": "chat",
            "message": message,
            "goal": {"type": "respond", "priority": "companionship",
                     "source": "chat"},
            "proposed_by": "user",
        },
        expresser=expresser,
        **kw,
    )


def tool_action(action_type: str, *, executor: ActionExecutor,
                params: Mapping[str, Any] | None = None, **kw) -> TickRecord:
    """A specific thing to do, proposed from outside.

    A proposal, never a decision: it is selected at stage 7 and then handed to
    conscience at stage 6's arbitration like anything else, and can be refused or
    redirected there. That is the whole difference between a tool call that
    Saient made and a tool call made through her.
    """
    return tick(
        intent={
            "kind": "tool",
            "goal": {"type": action_type, "priority": "efficiency",
                     "source": "tool_request"},
            "params": dict(params or {}),
            "proposed_by": "external",
        },
        executor=executor,
        **kw,
    )
