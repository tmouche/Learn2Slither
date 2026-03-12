from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from utils.types import FloatT

@dataclass
class EnvironmentSettings(BaseModel):
    height:int
    width:int

@dataclass
class HyperParameters(BaseModel):
    discount: FloatT
    exploration_decay: FloatT
    exploration_rate: FloatT
    epoch: int
    learning_rate: FloatT
    max_action: int
@dataclass
class RewardSettings(BaseModel):
    green_apple: FloatT
    loose: FloatT
    noting: FloatT
    red_apple: FloatT
    win: FloatT
@dataclass
class AgentSettings(BaseModel):
    hyper_params: HyperParameters
    rewards: RewardSettings


@dataclass
class TrainingNew(BaseModel):
    env_params: EnvironmentSettings
    agent_params: AgentSettings