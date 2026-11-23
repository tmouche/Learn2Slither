from app.application import Application
from core.agent_dql import AgentDQL
from core.environment import Environment
from utils.dataclass import (
    AgentSettings,
    EnvironmentSettings,
)
from utils.enum import Mode

env_settings = EnvironmentSettings()
env_settings.width = 16
env_settings.height = 16
env = Environment(env_settings)
ag_settings = AgentSettings()
ag_settings.agent_name = "test_functionnal_dql_12_5k"
agent = AgentDQL(
    env=env,
    agent_settings=ag_settings,
    batch_size=10,
)
agent.load(
    "test_functionnal_dql_12_5k"
)

app = Application(
    game_env=env,
    agent=agent,
    mode=Mode.AI_PLAY
)
app.launch(session=1)