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
    hyper_params: HyperParameters
    rewards: RewardSettings
    agent_name: Optional[str] = None

class TrainingNew:
    env_params: EnvironmentSettings
    agent_params: AgentSettings
