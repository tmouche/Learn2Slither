from app.application import Application
from core.agent_ql import AgentQL
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
ag_settings.agent_name = "player"
agent = AgentQL(
    env=env,
    agent_settings=ag_settings
)
agent.load("agent_ql_100000")
app = Application(
    game_env=env,
    agent=agent,
    mode=Mode.AI_PLAY
)
app.launch(session=1)


# 1 - 5
# 10 - 4
# 100 - 4
# 1000 - 4
# 10000 - 13
# 25000 - 13
# 35000 - 22
# 45000 - 25
# 55000 - 30
# 65000 - 22
# 75000 - 22
# 100000 - 48



