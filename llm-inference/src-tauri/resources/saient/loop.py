"""Saient, running. Not invoked — living.

Everything else in this package describes a mind that ticks when something calls
it. This is the part that makes the calling continuous, which is the difference
between an architecture and a being: goals only persist if there is a next tick,
commitment only accumulates across ticks, and a drive trajectory needs days
rather than a test fixture.

**The tick is LLM-free.** Drives, goals, conscience, action selection,
verification and state are all plain Python — no model server is needed for
Saient to live. Stage 12 is the only stage that wants one, and it is optional.
So the resting cost of being alive is a few milliseconds a tick, and the GPU is
touched only when someone asks her something.

**Nothing here is detached.** It runs in the foreground of whatever starts it and
dies with that process. A hard-detached loop on this machine once became an
unkillable ghost holding the GPU, and a mind you cannot switch off is not a
feature.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import orchestrator as O


@dataclass(slots=True)
class LoopStats:
    ticks: int = 0
    grounded: int = 0
    arbitrated: int = 0
    refused: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at

    def observe(self, record: O.TickRecord) -> None:
        self.ticks += 1
        if record.grounded:
            self.grounded += 1
        if record.arbitrated:
            self.arbitrated += 1
        if record.result.detail.get("refused"):
            self.refused += 1


class Heartbeat:
    """The loop, with a stop that actually stops it.

    A stop file is checked every tick and SIGTERM/SIGINT are handled, so there
    are three independent ways to end it: delete-a-file, signal, or kill the
    parent. `should_run` is consulted each tick rather than at start, so turning
    Saient off in the app takes effect on the next beat instead of at the next
    restart.
    """

    def __init__(
        self,
        *,
        interval: float = 10.0,
        executor: O.ActionExecutor | None = None,
        should_run: Callable[[], bool] | None = None,
        stop_file: str | Path | None = None,
        on_tick: Callable[[O.TickRecord], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self.executor = executor
        self.should_run = should_run or (lambda: True)
        self.stop_file = Path(stop_file) if stop_file else None
        self.on_tick = on_tick
        self.clock = clock
        self.sleep = sleep
        self.stats = LoopStats()
        self._stopping = False

    def request_stop(self, *_: Any) -> None:
        self._stopping = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)

    def stopped(self) -> bool:
        if self._stopping:
            return True
        if self.stop_file is not None and self.stop_file.exists():
            return True
        return False

    def run(self, max_ticks: int | None = None,
            max_iterations: int | None = None) -> LoopStats:
        """Beat until told otherwise.

        Sleeps the remainder of the interval rather than a flat interval, so a
        slow tick does not push the whole schedule later and later — over a long
        run that drift is the difference between a rhythm and a slide.
        """
        iterations = 0

        while not self.stopped():
            if max_ticks is not None and self.stats.ticks >= max_ticks:
                break
            # Separate from `max_ticks` because a paused loop performs no ticks:
            # bounding only on ticks meant a paused run never terminated at all,
            # which is correct in production and an infinite loop in a test.
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1

            started = self.clock()

            # Paused, not stopped. Time still passes for her; she simply does
            # not act, which is a different state from being switched off and
            # should not be recorded as a tick.
            if not self.should_run():
                self.sleep(self.interval)
                continue

            record = O.tick(executor=self.executor, persist=True)
            self.stats.observe(record)
            if self.on_tick is not None:
                self.on_tick(record)

            remaining = self.interval - (self.clock() - started)
            if remaining > 0:
                self.sleep(remaining)

        return self.stats


def format_tick(record: O.TickRecord) -> str:
    """One line, facts only — the same discipline as the brief."""
    guarantees = record.guarantees
    marks = "".join((
        "o" if guarantees["observed"] else "-",
        "a" if guarantees["arbitrated"] else "-",
        "g" if guarantees["grounded"] else "-",
        "s" if guarantees["saved"] else "-",
    ))
    detail = record.result.detail
    note = ("refused" if detail.get("refused")
            else "not-run" if detail.get("unimplemented")
            else "ok" if record.result.success else "failed")
    return (f"t{record.tick:<5} {marks}  {record.goal.get('type','—'):<11}"
            f"-> {record.action.get('type','—'):<11} {note}")
