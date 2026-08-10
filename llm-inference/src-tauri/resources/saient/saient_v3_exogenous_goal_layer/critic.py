from __future__ import annotations

from .schema import CriticDecision


class IndependentCritic:
    def __init__(
        self,
        novelty_weight: float = 0.8,
        reward_weight: float = 0.2,
        novelty_floor: float = 0.35,
        min_score: float = 0.45,
        reward_domination_ratio: float = 1.5,
    ) -> None:
        self.novelty_weight = float(novelty_weight)
        self.reward_weight = float(reward_weight)
        self.novelty_floor = float(novelty_floor)
        self.min_score = float(min_score)
        self.reward_domination_ratio = float(reward_domination_ratio)

    def evaluate(self, novelty_score: float, reward_score: float, proposal_origin: str) -> CriticDecision:
        novelty = float(max(0.0, min(1.0, novelty_score)))
        reward = float(max(0.0, min(1.0, reward_score)))
        score = self.novelty_weight * novelty + self.reward_weight * reward
        dominated = reward > (novelty * self.reward_domination_ratio)

        if proposal_origin == "inherited_system":
            return CriticDecision(
                accept=False,
                reason="rejected_inherited_system_goal",
                critic_score=score,
                novelty_score=novelty,
                reward_score=reward,
                dominated_by_reward=dominated,
            )
        if novelty < self.novelty_floor:
            return CriticDecision(
                accept=False,
                reason="rejected_low_novelty",
                critic_score=score,
                novelty_score=novelty,
                reward_score=reward,
                dominated_by_reward=dominated,
            )
        if dominated:
            return CriticDecision(
                accept=False,
                reason="rejected_reward_dominated",
                critic_score=score,
                novelty_score=novelty,
                reward_score=reward,
                dominated_by_reward=dominated,
            )
        if score < self.min_score:
            return CriticDecision(
                accept=False,
                reason="rejected_low_critic_score",
                critic_score=score,
                novelty_score=novelty,
                reward_score=reward,
                dominated_by_reward=dominated,
            )
        return CriticDecision(
            accept=True,
            reason="accepted_novelty_independent",
            critic_score=score,
            novelty_score=novelty,
            reward_score=reward,
            dominated_by_reward=dominated,
        )
