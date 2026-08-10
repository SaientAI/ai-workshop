from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "0.1.0"

try:
    from pydantic import BaseModel, Field

    class RuleBelief(BaseModel):
        rule_id: str
        score: float = 0.0
        usage_count: int = 0
        last_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


    class FailureLog(BaseModel):
        rule_id: str
        reason: str
        count: int = 1


    class SystemState(BaseModel):
        schema_version: str = SCHEMA_VERSION
        beliefs: list[RuleBelief] = Field(default_factory=list)
        failures: list[FailureLog] = Field(default_factory=list)
        meta: dict[str, Any] = Field(default_factory=dict)

except Exception:
    if os.environ.get("ARC_GREMLIN_STRICT_SCHEMA", "0") == "1":
        raise

    class _CompatBase:
        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

    class RuleBelief(_CompatBase):
        def __init__(self, rule_id: str, score: float = 0.0, usage_count: int = 0, last_used: datetime | None = None):
            self.rule_id = str(rule_id)
            self.score = float(score)
            self.usage_count = int(usage_count)
            self.last_used = last_used or datetime.now(timezone.utc)

    class FailureLog(_CompatBase):
        def __init__(self, rule_id: str, reason: str, count: int = 1):
            self.rule_id = str(rule_id)
            self.reason = str(reason)
            self.count = int(count)

    class SystemState(_CompatBase):
        def __init__(self, schema_version: str = SCHEMA_VERSION, beliefs: list[RuleBelief] | None = None, failures: list[FailureLog] | None = None, meta: dict[str, Any] | None = None):
            self.schema_version = str(schema_version)
            self.beliefs = list(beliefs or [])
            self.failures = list(failures or [])
            self.meta = dict(meta or {})


def migrate_state(data: dict[str, Any]) -> dict[str, Any]:
    d = dict(data or {})
    version = str(d.get("schema_version", "0.0.0"))

    if version == "0.0.0":
        d.setdefault("beliefs", [])
        d.setdefault("failures", [])
        d.setdefault("meta", {})
        d["schema_version"] = SCHEMA_VERSION
        version = SCHEMA_VERSION

    if version != SCHEMA_VERSION:
        d.setdefault("beliefs", [])
        d.setdefault("failures", [])
        d.setdefault("meta", {})
        d["schema_version"] = SCHEMA_VERSION

    return d
