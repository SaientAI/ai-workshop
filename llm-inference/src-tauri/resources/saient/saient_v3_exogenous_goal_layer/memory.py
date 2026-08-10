from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import PersistedGoal


class SelfMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, PersistedGoal] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._items = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._items = {}
            return
        out: dict[str, PersistedGoal] = {}
        for row in raw if isinstance(raw, list) else []:
            if not isinstance(row, dict):
                continue
            obj = str(row.get("objective", ""))
            if not obj:
                continue
            out[obj] = PersistedGoal(
                goal_id=str(row.get("goal_id", f"mem-{obj}")),
                objective=obj,
                first_seen_tick=int(row.get("first_seen_tick", 0)),
                last_seen_tick=int(row.get("last_seen_tick", 0)),
                streak=int(row.get("streak", 1)),
                reassertions=int(row.get("reassertions", 0)),
                last_status=str(row.get("last_status", "accepted")),
            )
        self._items = out

    def save(self) -> None:
        payload = [asdict(v) for v in sorted(self._items.values(), key=lambda x: (x.last_seen_tick, x.objective))]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def objectives(self) -> list[str]:
        return list(self._items.keys())

    def top(self) -> PersistedGoal | None:
        if not self._items:
            return None
        return max(self._items.values(), key=lambda p: (p.streak, p.last_seen_tick))

    def upsert_accept(self, goal_id: str, objective: str, tick: int, reasserted: bool) -> PersistedGoal:
        obj = str(objective)
        if obj in self._items:
            p = self._items[obj]
            p.last_seen_tick = int(tick)
            p.streak += 1
            if reasserted:
                p.reassertions += 1
            p.last_status = "accepted"
            return p
        p = PersistedGoal(
            goal_id=str(goal_id),
            objective=obj,
            first_seen_tick=int(tick),
            last_seen_tick=int(tick),
            streak=1,
            reassertions=1 if reasserted else 0,
            last_status="accepted",
        )
        self._items[obj] = p
        return p
