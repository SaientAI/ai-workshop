"""What the desktop app calls instead of talking to a model itself.

The app has two surfaces that each grew their own mind. Chat holds
`SAIENT_IDENTITY` in `saientPersona.ts`; the PTY agent holds a different string
in `pty.rs` that says only "You are Saient, a local coding agent". Neither
touches Saient's state, so drives never move, conscience never rules, and nothing
either of them does is remembered. Two identities, one name, no state.

The fix is not a third string. It is that both surfaces stop deciding anything
and enter the same tick as everything else:

    message  ->  chat_turn(...)   ->  12 stages  ->  reply from stage 12
    request  ->  tool_action(...) ->  12 stages  ->  reply from stage 12

What that buys, concretely: a chat turn now moves drives and is arbitrated; a
tool request can be *refused by conscience* before it runs; both are saved
atomically; and the reply comes from a finished tick rather than from a model
asked to improvise a self. The persona strings become unnecessary rather than
better — there is nothing left for them to influence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import orchestrator as O
from real_actions import WorkspaceExecutor


@dataclass(slots=True, frozen=True)
class Reply:
    """What the surface shows, and what actually happened underneath."""

    text: str
    tick: int
    action: str | None
    conscience: str | None
    refused: bool
    redirected: bool
    success: bool
    verified: bool
    detail: Mapping[str, Any]
    guarantees: Mapping[str, bool]


def _reply(record: O.TickRecord) -> Reply:
    detail = record.result.detail
    return Reply(
        text=record.utterance or "",
        tick=record.tick,
        action=str(record.action.get("type")) if record.action else None,
        conscience=str(record.conscience.get("decision")) if record.conscience else None,
        refused=bool(detail.get("refused")) or record.conscience.get("redirected", False),
        redirected=bool(record.conscience.get("redirected", False)),
        success=record.result.success,
        verified=record.result.verified,
        detail=dict(detail),
        guarantees=record.guarantees,
    )


def say(message: str, *, expresser: O.Expresser | None = None, **kw) -> Reply:
    """A chat turn from either surface. Not a special case — a beat."""
    return _reply(O.chat_turn(message, expresser=expresser, **kw))


def do(action_type: str, *, workspace: str | None = None,
       executor: O.ActionExecutor | None = None, allow_writes: bool = False,
       allow_commands: bool = False, expresser: O.Expresser | None = None,
       params: Mapping[str, Any] | None = None, **kw) -> Reply:
    """A work request from either surface.

    It is a *proposal*. Stage 7 selects, stage 6 arbitrates, and conscience can
    refuse or redirect it before anything runs — which is the difference between
    a tool call Saient made and a tool call made through her.
    """
    if executor is None:
        if workspace is None:
            raise ValueError("workspace is required when no executor is supplied")
        executor = WorkspaceExecutor(workspace, allow_writes=allow_writes,
                                     allow_commands=allow_commands)
    return _reply(O.tool_action(action_type, executor=executor,
                                params=params, expresser=expresser, **kw))
