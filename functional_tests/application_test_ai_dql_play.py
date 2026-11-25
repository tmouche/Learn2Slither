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
ag_settings.agent_name = "player"
agent = AgentDQL(
    env=env,
    agent_settings=ag_settings,
    batch_size=10,
)
agent.load(
    "agent_dql500"
)

app = Application(
    game_env=env,
    agent=agent,
    mode=Mode.AI_PLAY
)
app.launch(session=1)

# 1 - 3
# 10 - 2
# 100 - 2
# 1000 - 4
# 1500 - 26
# 2000 - 28
# 2500 - 34
# 3000 - 43
# 3500 - 48
