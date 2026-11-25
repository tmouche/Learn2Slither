from app.application import Application
from core.agent_dql import AgentDQL
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
hp_settings = HyperParameters()
rw_settings.green_apple = 0.5
rw_settings.red_apple = -0.5
rw_settings.noting = -0.02
rw_settings.loose = -1.0
rw_settings.win = 1.0

hp_settings.discount = 0.95
hp_settings.exploration_rate = 1.
hp_settings.learning_rate = 2.37e-03
hp_settings.max_action = 10000
hp_settings.epoch = 3000
hp_settings.exploration_min = 0.05
hp_settings.exploration_frac = (hp_settings.exploration_rate - hp_settings.exploration_min) / (0.45 * hp_settings.epoch)

agg_settings = AgentSettings()
agg_settings.agent_name = "agent_dql_3000"
agg_settings.hyper_params = hp_settings
agg_settings.rewards = rw_settings

agent = AgentDQL(env=env, agent_settings=agg_settings, batch_size=10)

# app = Application(game_env=env, mode=Mode.AI_TRAIN, agent=agent)
# app.launch(session=1)
agent.train()
agent.save()
agent.save_training_metrics()
