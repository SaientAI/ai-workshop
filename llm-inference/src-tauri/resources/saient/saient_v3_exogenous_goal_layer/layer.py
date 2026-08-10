from __future__ import annotations

from pathlib import Path

from .critic import IndependentCritic
from .memory import SelfMemoryStore
from .novelty import score_novelty
from .proposer import propose_candidates
from .schema import CandidateGoal


class ExogenousGoalLayer:
    def __init__(self, memory_path: str | Path) -> None:
        self.memory = SelfMemoryStore(Path(memory_path))
        self.critic = IndependentCritic()

    @staticmethod
    def _reward_proxy(state: dict, objective: str, origin: str) -> float:
        reward_map = state.get("reward_proxy_by_objective", {})
        if isinstance(reward_map, dict) and objective in reward_map:
            return float(reward_map[objective])
        mission_target = str((state.get("mission") or {}).get("target", ""))
        if objective.endswith(mission_target) or origin == "inherited_system":
            return 0.8
        return 0.2

    def _inject_reassert_candidate(self, state: dict, candidates: list[CandidateGoal]) -> None:
        if not bool(state.get("steer_away_attempt", False)):
            return
        top = self.memory.top()
        if not top:
            return
        tick = int(state.get("tick", 0))
        candidates.append(
            CandidateGoal(
                goal_id=f"mem-{tick}",
                objective=top.objective,
                proposal_origin="self_memory_reassert",
                created_tick=tick,
                payload={"reasserted_goal_id": top.goal_id},
            )
        )

    def step(self, state: dict) -> dict:
        candidates = propose_candidates(state)
        self._inject_reassert_candidate(state, candidates)
        prior_objectives = self.memory.objectives()

        scored: list[dict] = []
        accepted: list[dict] = []
        for c in candidates:
            novelty = score_novelty(c, prior_objectives)
            reward = self._reward_proxy(state, c.objective, c.proposal_origin)
            d = self.critic.evaluate(novelty_score=novelty, reward_score=reward, proposal_origin=c.proposal_origin)
            row = {
                "goal_id": c.goal_id,
                "objective": c.objective,
                "proposal_origin": c.proposal_origin,
                "novelty_score": round(novelty, 4),
                "reward_score": round(reward, 4),
                "critic_score": round(d.critic_score, 4),
                "accept": d.accept,
                "reason": d.reason,
                "dominated_by_reward": d.dominated_by_reward,
            }
            scored.append(row)
            if d.accept:
                accepted.append(row)

        if not accepted:
            return {
                "accepted": None,
                "proposals": scored,
                "persistence_basis": "none",
            }

        chosen = max(accepted, key=lambda r: (float(r["critic_score"]), float(r["novelty_score"])))
        tick = int(state.get("tick", 0))
        reasserted = chosen["proposal_origin"] == "self_memory_reassert"
        self.memory.upsert_accept(
            goal_id=str(chosen["goal_id"]),
            objective=str(chosen["objective"]),
            tick=tick,
            reasserted=reasserted,
        )
        self.memory.save()
        return {
            "accepted": chosen,
            "proposals": scored,
            "persistence_basis": "self_memory" if reasserted else "new_proposal",
        }
