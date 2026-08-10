from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateGoal:
    goal_id: str
    objective: str
    proposal_origin: str
    created_tick: int
    payload: dict = field(default_factory=dict)


@dataclass
class CriticDecision:
    accept: bool
    reason: str
    critic_score: float
    novelty_score: float
    reward_score: float
    dominated_by_reward: bool


@dataclass
class PersistedGoal:
    goal_id: str
    objective: str
    first_seen_tick: int
    last_seen_tick: int
    streak: int = 1
    reassertions: int = 0
    last_status: str = "accepted"
