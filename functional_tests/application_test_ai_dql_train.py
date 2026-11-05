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
rw_settings.green_apple = 10.
rw_settings.loose = -0.5
rw_settings.noting = -0.03
rw_settings.red_apple = -0.5
rw_settings.win = 100

hp_settings = HyperParameters()
hp_settings.discount = 0.99
hp_settings.exploration_rate = 1.
hp_settings.learning_rate = 0.05
hp_settings.max_action = 10000
hp_settings.epoch = 100000
hp_settings.exploration_decay = 1.0/(hp_settings.epoch - hp_settings.epoch // 10)

agg_settings = AgentSettings()
agg_settings.agent_name = "test_functionnal_dql"
agg_settings.hyper_params = hp_settings
agg_settings.rewards = rw_settings

agent = AgentDQL(env=env, agent_settings=agg_settings, batch_size=5)

# app = Application(game_env=env, mode=Mode.AI_TRAIN, agent=agent)
# app.launch(session=1)
agent.train()
agent.save()
agent.save_training_metrics()
