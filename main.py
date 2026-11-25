from app.application import Application
from utils.logger import logger
from core.agent import Agent
from core.agent_ql import AgentQL
from core.agent_dql import AgentDQL
from core.environment import Environment
from utils.dataclass import (
    EnvironmentSettings,
    AgentSettings,
    HyperParameters,
    RewardSettings
)
from utils.enum import Mode
from utils.settings import Settings
import sys


def environment_settings() -> EnvironmentSettings:
    sett = EnvironmentSettings()
    sett.height = 10
    sett.width = 10
    return sett


def hyper_parameters(session: int) -> HyperParameters:
    hp = HyperParameters()
    hp.discount = 0.95
    hp.exploration_rate = 1.0
    hp.learning_rate = 0.2
    hp.max_action = 10000
    hp.epoch = session
    hp.exploration_min = 0.05
    hp.exploration_frac = (
        (
            hp.exploration_rate -
            hp.exploration_min
        ) / (0.3 * session)
    )
    return hp


def reward_settings() -> RewardSettings:
    rewards = RewardSettings()
    rewards.green_apple = 10.0
    rewards.loose = -100.0
    rewards.noting = -1
    rewards.red_apple = -11.0
    rewards.win = 100.0
    return rewards


def agent_settings(
    training_name: str,
    hyper_parameters: HyperParameters,
    reward_settings: RewardSettings,
) -> AgentSettings:
    agg = AgentSettings()
    agg.agent_name = training_name
    agg.hyper_params = hyper_parameters
    agg.rewards = reward_settings
    return agg


def play(env: Environment, agent: Agent, app_mode: Mode = Mode.HUMAIN_PLAY):
    logger.info("Starting application...")
    app = Application(env, agent, mode=app_mode)
    app.launch()


def _main_load(settings: Settings, agent: Agent):
    try:
        agent.load(settings.load_file)
    except Exception:
        logger.error("Exit Learn2Slither..")
        sys.exit(1)


def _main_train(settings: Settings, agent: Agent, env: Environment):
    if not settings.visual:
        agent.train()
    else:
        play(env, agent, Mode.AI_TRAIN)
    agent.save()
    agent.save_training_metrics()


def _main_play_loaded(
    settings: Settings,
    agent: Agent,
    env: Environment,
):
    if settings.visual:
        play(env, agent, Mode.AI_PLAY)
    else:
        agent._evaluation(settings.session)


def main():
    settings = Settings()
    settings.parse()
    sett = environment_settings()
    hp = hyper_parameters(settings.session)
    rs = reward_settings()
    agg = agent_settings(
        training_name=settings.train_file,
        hyper_parameters=hp,
        reward_settings=rs
    )
    env = Environment(sett)
    if settings.agent_type == settings.QL:
        agent = AgentQL(env=env, agent_settings=agg)
    else:
        agent = AgentDQL(env=env, agent_settings=agg, batch_size=15)
    if settings.load:
        _main_load(settings, agent)
    if settings.train:
        _main_train(settings, agent, env)
    elif settings.load:
        _main_play_loaded(settings, agent, env)
    else:
        if settings.visual:
            play(env, agent)
        else:
            logger.warning("Nothing to do!")


if __name__ == "__main__":
    main()
