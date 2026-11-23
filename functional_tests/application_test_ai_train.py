from app.application import Application
from core.agent_ql import AgentQL
from core.environment import Environment
from utils.dataclass import (
    AgentSettings,
    EnvironmentSettings,
    HyperParameters,
    RewardSettings,
)
from utils.enum import Mode

env_settings = EnvironmentSettings()
env_settings.width = 10
env_settings.height = 10
env = Environment(env_settings)

rw_settings = RewardSettings()
rw_settings.green_apple = 0.5
rw_settings.red_apple = -1
rw_settings.noting = -0.02
rw_settings.loose = -5.0
rw_settings.win = 1.0

hp_settings = HyperParameters()
hp_settings.discount = 0.95
hp_settings.exploration_rate = 1.
hp_settings.learning_rate = 0.05
hp_settings.max_action = 1000
hp_settings.epoch = 100000
hp_settings.exploration_min = 0.02
hp_settings.exploration_frac = (hp_settings.exploration_rate - hp_settings.exploration_min) / (0.3 * hp_settings.epoch)

agg_settings = AgentSettings()
agg_settings.agent_name = "test_agent_100k"
agg_settings.hyper_params = hp_settings
agg_settings.rewards = rw_settings

agent = AgentQL(env=env, agent_settings=agg_settings)

# app = Application(game_env=env, mode=Mode.AI_TRAIN, agent=agent)
# app.launch(session=1)

agent.train()
agent.save()
agent.save_training_metrics()
