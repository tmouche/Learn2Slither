from app.application import Application
from core.environment import Environment
from utils.dataclass import (
    EnvironmentSettings
)

env_settings = EnvironmentSettings()
env_settings.width = 10
env_settings.height = 10
env = Environment(env_settings)
app = Application(game_env=env)
app.launch()

