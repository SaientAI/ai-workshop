"""Exclusive sidecar locks shared by desktop runtime writers on both platforms."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time


_WINDOWS = os.name == "nt"


@contextmanager
def exclusive_file_lock(path: str | Path):
    """Hold a persistent sidecar lock; never continue without acquiring it.

    Windows locks byte zero, even on an empty file (supported by msvcrt).
    Nonblocking attempts let contention wait as long as POSIX flock rather
    than failing after LK_LOCK's ten one-second attempts. No lock-file writes,
    truncation or removal occur, so cooperating writers retain one lock identity.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        if _WINDOWS:
            import msvcrt

            while True:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
