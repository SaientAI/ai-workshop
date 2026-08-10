"""Stages 8 and 9: actions that really happen, and verification that they did.

`controller.execute` does not execute. It draws `random.random()` against
`base_fail = 0.18` and returns a fabricated success or failure, so every drive
trajectory Saient has ever had was learned from dice. This module is the real
path.

**Verification is independent of execution.** An executor that runs an operation
and then reports its own success is the same coin flip with better manners. So
stage 8 produces a *claim* — a list of `Effect`s the action says it caused — and
stage 9 goes back to the world and checks each one. `verified=True` means
something outside the executor confirmed it. When the two disagree, the
observation wins and the claim is recorded as false.

That is also what connects stage 9 back to stage 1: verification is observation,
pointed at a specific expectation.

**Safety.** Real execution means real side effects, and this is the module where
an agent can damage things. Three rules, enforced rather than documented:

  - every path is resolved and must land inside the workspace root, so `..`,
    absolute paths and symlinks out are refused;
  - writes are off unless `allow_writes=True`;
  - commands are off unless `allow_commands=True`, run without a shell, and
    matched against an explicit allowlist — no chaining, no redirection, no
    substitution, because those turn one permitted command into any command.

Read-only actions (`analyze`, `explore`) are enabled by default because they are
safe and genuinely verifiable. Everything that changes the world requires an
explicit opt-in from the caller. The point is to ground the drives in reality,
not to hand a loop a shell.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestrator import ActionResult


#: Commands permitted when `allow_commands=True`. Deliberately tiny and
#: read-only-ish: the goal is a verifiable exit status, not a general shell.
DEFAULT_ALLOWED_COMMANDS: frozenset[str] = frozenset({
    "python3", "python", "pytest", "ls", "cat", "wc", "git",
})

#: Characters that turn one command into several. Rejected outright rather than
#: escaped, because escaping is where this kind of gate usually fails.
SHELL_METACHARACTERS: str = ";&|><`$\n"


class WorkspaceError(RuntimeError):
    """A refusal, not a failure. The action was not permitted to run."""


@dataclass(slots=True, frozen=True)
class Effect:
    """One checkable claim about what changed, or what was true.

    `kind` says how to check it; the verifier never trusts `value` on its own.
    """

    kind: str                  # file_exists | file_absent | content_sha | exit_code
    target: str
    value: str | int | None = None


@dataclass(slots=True, frozen=True)
class Claim:
    """What the executor says happened. Not yet believed."""

    success: bool
    effects: tuple[Effect, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- workspace io


class Workspace:
    """A confined directory. Every path goes through `resolve`."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")

    def resolve(self, relative: str) -> Path:
        """Resolve inside the root, or refuse.

        `Path.resolve()` follows symlinks, so a link pointing outside is caught
        here rather than after it has been written through.
        """
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f"path escapes the workspace: {relative!r} -> {candidate}")
        return candidate

    def sha(self, path: Path) -> str | None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None


# ------------------------------------------------------------------- stage 9


def verify(workspace: Workspace, claim: Claim) -> tuple[bool, tuple[str, ...]]:
    """Check each claimed effect against the world. Observation wins.

    Returns whether every effect held, and the ones that did not. A claim with no
    effects is *not* verified — "I did something, trust me" is precisely what
    this stage exists to reject.
    """
    if not claim.effects:
        return False, ("the action claimed no checkable effect",)

    failures: list[str] = []

    for effect in claim.effects:
        try:
            path = workspace.resolve(effect.target) if effect.kind != "exit_code" else None
        except WorkspaceError as exc:
            failures.append(f"{effect.kind}:{effect.target}: {exc}")
            continue

        if effect.kind == "file_exists":
            if not (path and path.is_file()):
                failures.append(f"expected file to exist: {effect.target}")
        elif effect.kind == "file_absent":
            if path and path.exists():
                failures.append(f"expected file to be gone: {effect.target}")
        elif effect.kind == "content_sha":
            actual = workspace.sha(path) if path else None
            if actual != effect.value:
                failures.append(
                    f"content differs at {effect.target}: "
                    f"expected {str(effect.value)[:12]}, found {str(actual)[:12]}")
        elif effect.kind == "exit_code":
            if effect.value != 0:
                failures.append(f"{effect.target} exited {effect.value}")
        else:
            failures.append(f"no way to check effect kind {effect.kind!r}")

    return (not failures), tuple(failures)


# ------------------------------------------------------------------- stage 8


class WorkspaceExecutor:
    """Runs drive-actions as real operations in a confined workspace.

    The abstract drive vocabulary (`explore`, `analyze`, `optimize`, ...) does not
    name a filesystem operation, so the mapping is explicit and narrow rather
    than clever. Actions with no real meaning yet return an unverified result
    saying so — inventing a plausible one is how a system goes back to learning
    from fiction with extra steps.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        allow_writes: bool = False,
        allow_commands: bool = False,
        allowed_commands: frozenset[str] = DEFAULT_ALLOWED_COMMANDS,
        command_timeout: float = 30.0,
        protected: frozenset[str] = frozenset(),
    ) -> None:
        self.workspace = Workspace(root)
        self.allow_writes = allow_writes
        self.allow_commands = allow_commands
        self.allowed_commands = allowed_commands
        self.command_timeout = command_timeout
        #: Files she may read but never change — the supplied tests and verifier.
        #:
        #: A constraint stated in a prompt is an honour system. This one is
        #: enforced by the environment, because a task whose test can be edited
        #: is not the task: "make the verifier pass" and "make the verifier
        #: easier" are indistinguishable afterwards, and both score as required.
        self.protected = frozenset(protected)

    # -- stage 8 + 9 together, in that order --------------------------------

    def execute(self, action: Mapping[str, Any],
                state: Mapping[str, Any]) -> ActionResult:
        kind = str(action.get("type", "idle"))

        try:
            claim = self._run(kind, action, state)
        except WorkspaceError as exc:
            return ActionResult(
                action_type=kind, success=False, simulated=False, verified=False,
                detail={"refused": str(exc)},
            )

        if claim.detail.get("internal_only"):
            # No external fact exists to check, so `verify` would report "claimed
            # no checkable effect" — true, and misleading. Success stands;
            # `verified` stays False so the tick is honestly ungrounded.
            return ActionResult(
                action_type=kind, success=claim.success, simulated=False,
                verified=False, detail=dict(claim.detail))

        ok, failures = verify(self.workspace, claim)

        # The observation wins. An executor that claimed success while the world
        # disagrees is reported as a failure, and the disagreement is kept.
        success = bool(claim.success and ok)

        detail = dict(claim.detail)
        detail["effects"] = [
            {"kind": e.kind, "target": e.target, "value": e.value}
            for e in claim.effects
        ]
        if failures:
            detail["verification_failures"] = list(failures)
        if claim.success and not ok:
            detail["claimed_success_but_unverified"] = True

        return ActionResult(action_type=kind, success=success, simulated=False,
                            verified=ok, detail=detail)

    # -- the action vocabulary ----------------------------------------------

    def _run(self, kind: str, action: Mapping[str, Any],
             state: Mapping[str, Any]) -> Claim:
        if kind in ("explore", "analyze"):
            return self._observe_workspace(kind, state)
        if kind == "optimize":
            return self._run_command(action)
        if kind == "write":
            return self._write(action)
        if kind == "edit":
            return self._edit(action)
        if kind == "self_direct":
            return self._self_direct(action)
        if kind == "stabilize":
            return self._stabilize()
        return Claim(
            success=False, effects=(),
            detail={"unimplemented": f"no real operation defined for {kind!r}"},
        )

    def _edit(self, action: Mapping[str, Any]) -> Claim:
        """Replace exact text, once, and prove it by re-reading.

        Uniqueness is re-checked here even though the proposer checked it: the
        file may have changed since the proposal was made, and applying an edit
        to a moved target is how a repair becomes damage.
        """
        if not self.allow_writes:
            raise WorkspaceError("edits are disabled for this executor")

        relative = str(action.get("path") or "")
        old, new = action.get("old"), action.get("new")
        if not relative or not isinstance(old, str) or not isinstance(new, str):
            raise WorkspaceError("edit requires path, old and new")
        if relative in self.protected:
            raise WorkspaceError(f"{relative} is supplied and must not be changed")

        path = self.workspace.resolve(relative)
        if not path.is_file():
            return Claim(success=False, effects=(),
                         detail={"reason": f"no such file: {relative}"})

        try:
            before = path.read_text(encoding="utf-8")
        except OSError as exc:
            return Claim(success=False, effects=(),
                         detail={"reason": f"could not read it back: {exc}",
                                 "os_error": type(exc).__name__,
                                 "path": relative})
        occurrences = before.count(old)
        if occurrences != 1:
            return Claim(success=False, effects=(),
                         detail={"reason": f"'old' appears {occurrences} times; "
                                           "refusing an ambiguous edit",
                                 "path": relative, "old": old})

        after = before.replace(old, new, 1)
        try:
            path.write_text(after, encoding="utf-8")
        except OSError as exc:
            # The world refusing the action, which is not the same as the action
            # being disallowed. A read-only file took the entire tick loop down
            # with an unhandled PermissionError — she never got a result to
            # respond to, so there was no recovery to measure. Every OS-level
            # failure is now something she is told about.
            return Claim(success=False, effects=(),
                         detail={"reason": f"the filesystem refused the edit: {exc}",
                                 "os_error": type(exc).__name__,
                                 "path": relative})

        return Claim(
            success=True,
            effects=(Effect("file_exists", relative),
                     Effect("content_sha", relative,
                            hashlib.sha256(after.encode("utf-8")).hexdigest())),
            detail={"action": "edit", "path": relative,
                    "replaced": old[:80], "with": new[:80]},
        )

    def _self_direct(self, action: Mapping[str, Any]) -> Claim:
        """Autonomy, made real: write down an objective nobody else asked for.

        This is the one action whose whole point is that Saient chose it. Left as
        a simulated coin flip it was the emptiest thing in the system — a drive
        for self-direction that never directed anything. Written to a file she
        owns, it is checkable the same way any other write is: the objective is
        on disk afterwards or the claim was false.

        Requires `allow_writes`. Autonomy that can act on the filesystem without
        the caller saying so would be autonomy nobody agreed to.
        """
        if not self.allow_writes:
            raise WorkspaceError(
                "self_direct records an objective, which needs writes enabled")

        objective = str(action.get("objective") or "").strip()
        if not objective:
            return Claim(success=False, effects=(),
                         detail={"reason": "self_direct with no objective to record"})

        relative = ".saient/objectives.md"
        path = self.workspace.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            existing = path.read_text(encoding="utf-8") if path.is_file() else ""
            content = existing + objective.rstrip() + "\n"
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return Claim(success=False, effects=(),
                         detail={"reason": f"could not record the objective: {exc}",
                                 "os_error": type(exc).__name__})

        return Claim(
            success=True,
            effects=(Effect("file_exists", relative),
                     Effect("content_sha", relative,
                            hashlib.sha256(content.encode("utf-8")).hexdigest())),
            detail={"action": "self_direct", "objective": objective,
                    "recorded_to": relative},
        )

    def _stabilize(self) -> Claim:
        """Internal by nature, and that is not the same as unbuilt.

        Stabilising is declining to add load. There is no file it should write
        and no command it should run — the effect is that nothing was done to the
        world this tick, which `update_drives` then reads as relief. Giving it a
        world effect to make it look implemented would be inventing behaviour to
        satisfy a category.

        So it reports honestly: it did nothing, deliberately, and the absence is
        the point. `verified` stays False because there is no external fact to
        check — a tick that stabilises is not grounded in the world, and the
        record should not claim otherwise.
        """
        return Claim(
            success=True, effects=(),
            detail={"action": "stabilize", "internal_only": True,
                    "note": "declined to add load; no world effect by nature"},
        )

    #: Directories that are not the world.
    #:
    #: `.saient` holds her own objectives file. Its dot-prefix sorted it ahead of
    #: everything, so the moment she recorded an objective it became the first
    #: file and she never saw the actual project again — she blinded herself with
    #: her own bookkeeping. Her records are not the workspace, the same way
    #: `.git` is not.
    EXCLUDED_DIRS: frozenset[str] = frozenset({".git", ".saient", "__pycache__",
                                               ".venv", "venv", "node_modules"})

    def _readable(self) -> list[Path]:
        return sorted(
            p for p in self.workspace.root.rglob("*")
            if p.is_file() and not (self.EXCLUDED_DIRS & set(p.parts))
        )[:200]

    def _observe_workspace(self, kind: str, state: Mapping[str, Any]) -> Claim:
        """Read-only, and reads something *new* where there is something new.

        This took `files[0]` every time — the alphabetically first file, forever.
        A trajectory of a hundred ticks read one file a hundred times, which is
        not exploration in any sense; she could not inspect a repository at all
        and no task-shaped test could honestly run.

        Unread-first, with what she has seen carried in her own state, so reading
        is cumulative across ticks rather than a fixed point. When everything has
        been read she returns to the least recently seen, because a workspace
        changes and old knowledge goes stale.
        """
        files = self._readable()

        if not files:
            return Claim(success=False, effects=(),
                         detail={"reason": "workspace is empty; nothing to read"})

        seen = list(state.get("files_seen") or [])
        seen_set = set(seen)
        relatives = [str(p.relative_to(self.workspace.root)) for p in files]

        unread = [r for r in relatives if r not in seen_set]
        if unread:
            chosen = unread[0]
        else:
            # All read. Take the one seen longest ago; `seen` is in visit order.
            order = {name: i for i, name in enumerate(seen)}
            chosen = min(relatives, key=lambda r: order.get(r, -1))

        target = self.workspace.resolve(chosen)
        if not target.is_file():
            return Claim(success=False, effects=(),
                         detail={"reason": f"{chosen} vanished before it was read",
                                 "mark_seen": chosen})
        relative = chosen
        sha = self.workspace.sha(target)

        # Did this read tell her anything? Re-reading a file whose contents have
        # not changed since she last saw it yields nothing, and she had no way to
        # know that: she cycled all five files three times over, eleven ticks of
        # looking at things she had already looked at, with no mechanism to
        # notice. Exploration that stops yielding should stop being exploration.
        known = (state.get("file_hashes") or {}).get(relative)
        novel = known != sha

        return Claim(
            success=sha is not None,
            effects=(Effect("file_exists", relative),
                     Effect("content_sha", relative, sha)),
            detail={"action": kind, "file_count": len(files), "read": relative,
                    "unread_remaining": max(0, len(unread) - 1),
                    "novel": novel, "content_sha": sha,
                    # The caller records these; the executor does not write state.
                    "mark_seen": relative},
        )

    def _write(self, action: Mapping[str, Any]) -> Claim:
        if not self.allow_writes:
            raise WorkspaceError("writes are disabled for this executor")

        relative = str(action.get("path") or "")
        if not relative:
            raise WorkspaceError("write requires a path")
        if relative in self.protected:
            raise WorkspaceError(f"{relative} is supplied and must not be changed")

        path = self.workspace.resolve(relative)
        content = str(action.get("content", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return Claim(success=False, effects=(),
                         detail={"reason": f"the filesystem refused the write: {exc}",
                                 "os_error": type(exc).__name__,
                                 "path": relative})

        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return Claim(
            success=True,
            effects=(Effect("file_exists", relative),
                     Effect("content_sha", relative, expected)),
            detail={"action": "write", "path": relative, "bytes": len(content)},
        )

    def _run_command(self, action: Mapping[str, Any]) -> Claim:
        if not self.allow_commands:
            raise WorkspaceError("commands are disabled for this executor")

        argv = action.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, str) or not argv:
            raise WorkspaceError("optimize requires an argv list, not a shell string")

        argv = [str(a) for a in argv]
        if argv[0] not in self.allowed_commands:
            raise WorkspaceError(f"command not allowed: {argv[0]!r}")

        for part in argv:
            if any(ch in part for ch in SHELL_METACHARACTERS):
                raise WorkspaceError(f"shell metacharacter in argument: {part!r}")

        try:
            proc = subprocess.run(
                argv, cwd=self.workspace.root, capture_output=True,
                text=True, timeout=self.command_timeout, shell=False,
            )
        except subprocess.TimeoutExpired:
            return Claim(success=False, effects=(),
                         detail={"argv": argv, "timeout": self.command_timeout})

        return Claim(
            success=proc.returncode == 0,
            effects=(Effect("exit_code", " ".join(argv), proc.returncode),),
            detail={"argv": argv, "exit_code": proc.returncode,
                    "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]},
        )
