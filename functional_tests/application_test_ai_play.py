from app.application import Application
from core.agent_ql import AgentQL
from core.environment import Environment
from utils.dataclass import (
    AgentSettings,
    EnvironmentSettings,
)
from utils.enum import Mode

env_settings = EnvironmentSettings()
env_settings.width = 20
env_settings.height = 20
env = Environment(env_settings)
ag_settings = AgentSettings()
ag_settings.agent_name = "test"
agent = AgentQL(
    env=env,
    agent_settings=ag_settings
)
agent.load("test_functionnal_100x100")
app = Application(
    game_env=env,
    agent=agent,
    mode=Mode.AI_PLAY
)
app.launch(session=2)


