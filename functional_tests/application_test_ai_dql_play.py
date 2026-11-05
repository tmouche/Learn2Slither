from app.application import Application
from core.agent_dql import AgentDQL
from core.environment import Environment
from utils.dataclass import (
    AgentSettings,
    EnvironmentSettings,
)
from utils.enum import Mode

env_settings = EnvironmentSettings()
env_settings.width = 10
env_settings.height = 10
env = Environment(env_settings)
ag_settings = AgentSettings()
ag_settings.agent_name = "test"
agent = AgentDQL(
    env=env,
    agent_settings=ag_settings,
    
)
app = Application(
    game_env=env,
    agent=agent,
    mode=Mode.AI_PLAY
)
app.launch(session=1)