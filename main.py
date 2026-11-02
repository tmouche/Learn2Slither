from app.application import Application
from argparse import ArgumentParser, Namespace
from typing import Any, Dict
from datetime import datetime
from utils.logger import logger
from core.agent_ql import AgentQL
from core.environment import Environment
from typing import Optional, List   
from utils.dataclass import EnvironmentSettings, AgentSettings, HyperParameters, RewardSettings
from utils.exception import FunctionalException
from utils.enum import Mode
import os

def get_args() -> Dict[str, Any]:
    parser = ArgumentParser(
        prog="snake",
        description="Reinforcement Learning project thought a snake game."
    )
    parser.add_argument(
        "-v",
        "--visual",
        help="Display in a Tkinter window.",
        dest="visual",
        nargs=1,
        default=["off"],
        type=str
    )
    parser.add_argument(
        "-t",
        "--train",
        help="Trigger a training session.",
        dest="training_name",
        nargs=1,
        type=str
    )
    parser.add_argument(
        "-l",
        "--load",
        help="Load a fully trained agent.",
        dest="load_file",
        nargs=1,
        type=str
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Plot differents metrics.",
        dest="plot",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--session",
        help="Number of sessions who will be played.",
        dest="session",
        nargs=1,
        type=int,
        default=[1],
    )
    return vars(parser.parse_args())

def get_visual(args: Dict[str, Any]) -> bool:
    visual: str = args.get("visual")[0]
    if visual not in ['on', 'off']:
        logger.error("Visual has to be either on or off")
        os._exit(1)
    return True if visual == 'on' else False

def get_training_name(args: Dict[str, Any]) -> str:
    training_name: Optional[List[str]] = args.get("training_name")
    if training_name is not None:
        training_name = training_name[0].strip()
    return training_name

def get_load_file(args: Dict[str, Any]) -> Optional[str]:
    load_file: Optional[List[str]] = args.get("load_file")
    if load_file is not None:
        load_file: str = load_file[0].strip()
    return load_file

def get_plot(args: Dict[str, Any]) -> bool:
    return args.get("plot")

def get_session(args: Dict[str, Any]) -> int:
    session: int = args.get("session")[0]
    if session <= 0:
        logger.error("Session should be striclty superior to 0")
        os._exit(1)
    return session

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
    hp.exploration_decay = 1.0/hp.epoch
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

def play(env: Environment, agent: AgentQL, app_mode: Mode = Mode.HUMAIN_PLAY):
    app = Application(env, agent, mode=app_mode)
    app.launch()

def main():
    args: Dict[str, Any] = get_args()
    sett = environment_settings()
    hp = hyper_parameters(get_session(args))
    rs = reward_settings()
    agg = agent_settings(
        training_name=get_training_name(args),
        hyper_parameters=hp,
        reward_settings=rs
    )
    env = Environment(sett)
    agent = AgentQL(env=env, agent_settings=agg)
    load_file: Optional[str] = get_load_file(args)
    if load_file is not None:
        agent.load(load_file)
    if get_training_name(args) is not None:
        if get_visual(args) == False:
            agent.train()
            agent.save()
            agent.save_training_metrics()
            if get_plot(args) == True:
                agent.plot_training_metrics()
        else:
            play(env, agent, Mode.AI_TRAIN)
    elif load_file:
        if get_visual(args) == True:
            play(env, agent, Mode.AI_PLAY)
    else:
        if get_visual(args) == True:
            play(env, agent)
        else:
            logger.warning("Nothing to do!")



if __name__ == "__main__":
    main()