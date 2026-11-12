from pydantic.dataclasses import dataclass
from typing import Optional
from utils.types import FloatT

class EnvironmentSettings:
    height:int
    width:int

class HyperParameters:
    discount: FloatT
    exploration_decay: FloatT
    exploration_rate: FloatT
    epoch: int
    learning_rate: FloatT
    max_action: int

class RewardSettings:
    green_apple: FloatT
    loose: FloatT
    noting: FloatT
    red_apple: FloatT
    win: FloatT

class AgentSettings:
    hyper_params: Optional[HyperParameters] = None
    rewards: Optional[RewardSettings] = None
    agent_name: str = "default"

class TrainingNew:
    env_params: EnvironmentSettings
    agent_params: AgentSettings

class Step:
    _epoch_step: int
    _accumulated_step: int

    def __init__(self):
        self._epoch_step = 0
        self._accumulated_step = 0

    @property
    def epoch_step(self) -> int:
        return self._epoch_step

    @property
    def accumulated_step(self) -> int:
        return self._accumulated_step

    def plus_one(self):
        self._epoch_step += 1
        self._accumulated_step += 1

    def reset_epoch(self):
        self._epoch_step = 0

    def reset(self):
        self.reset_epoch()
        self._accumulated_step = 0
