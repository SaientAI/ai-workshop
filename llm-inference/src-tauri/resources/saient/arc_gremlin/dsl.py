from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubsetRule:
    mode: str = "row"
    k: int = 3
    features: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    condition: dict[str, Any] | None = None
    order_sensitive: bool = True


@dataclass
class TransformRule:
    op: str


@dataclass
class ProgramRule:
    steps: list[Any] = field(default_factory=list)


@dataclass
class CompositeRule:
    feature: str
    op: str
    threshold: float
    rule_true: Any
    rule_false: Any
