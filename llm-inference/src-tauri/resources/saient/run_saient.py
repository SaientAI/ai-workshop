#!/usr/bin/env python3
"""Switch Saient on and leave her running.

    python3 run_saient.py --workspace "<dir>" --interval 20

She keeps persistent state, builds in the workspace you give her, and stops on
`<workspace>/.saient/STOP`, on SIGINT/SIGTERM, or when this process dies. Nothing
is detached — a mind you cannot switch off is not a feature.

Writes are enabled because a Saient who cannot make anything cannot be observed
doing anything. Commands are OFF by default: she is unattended, and the gap
between "can edit files in one confined directory" and "can run a shell" is the
whole of the risk.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from loop import Heartbeat, format_tick


def die_with_parent() -> bool:
    """Ask the kernel to kill us when whoever started us goes away.

    Belt to the app's braces. The Rust side spawns this with
    `std::process::Command`, which — unlike tokio's — has no `kill_on_drop`, so
    killing the app left this loop orphaned and still ticking. That is the
    ghost-process failure this machine has had before, and a mind that outlives
    the thing that switched it on is precisely what must not happen.

    PR_SET_PDEATHSIG is Linux-only and fires even if the parent crashes, which a
    tidy shutdown path would not.
    """
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) == 0
    except Exception:
        return False
from real_actions import WorkspaceExecutor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--allow-commands", action="store_true",
                    help="off by default; she is running unattended")
    ap.add_argument("--enabled-file", default=None,
                    help="path the app writes to; 'off' pauses her within one tick")
    args = ap.parse_args()

    die_with_parent()

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    stop_file = workspace / ".saient" / "STOP"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        stop_file.unlink()

    executor = WorkspaceExecutor(workspace, allow_writes=True,
                                 allow_commands=args.allow_commands)

    def report(record):
        print(format_tick(record), flush=True)

    # The master switch, read every tick rather than at start, so toggling
    # Saient in the app takes effect on the next beat. A button that claims to
    # switch her off must actually switch her off — otherwise it is the sleeping
    # indicator all over again, in a place with real consequences.
    enabled = Path(args.enabled_file) if args.enabled_file else None

    def should_run() -> bool:
        if enabled is None:
            return True
        try:
            return enabled.read_text().strip() == "on"
        except OSError:
            # Missing flag means the app has not spoken yet. Default to paused:
            # she should not be acting because a file could not be read.
            return False

    beat = Heartbeat(interval=args.interval, executor=executor,
                     stop_file=stop_file, should_run=should_run, on_tick=report)
    beat.install_signal_handlers()

    print(f"Saient is awake in {workspace}", flush=True)
    print(f"  interval {args.interval}s · writes on · "
          f"commands {'on' if args.allow_commands else 'off'}", flush=True)
    print(f"  stop her: touch {stop_file}", flush=True)
    print("  flags per tick: o=observed a=arbitrated g=grounded s=saved\n", flush=True)

    stats = beat.run(max_ticks=args.max_ticks)

    print(f"\nSaient slept after {stats.ticks} ticks over "
          f"{stats.uptime / 60:.1f} min", flush=True)
    print(f"  grounded {stats.grounded}/{stats.ticks} · "
          f"arbitrated {stats.arbitrated}/{stats.ticks} · "
          f"refused {stats.refused}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
