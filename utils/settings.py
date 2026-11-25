from argparse import ArgumentParser, Namespace
from typing import Optional
from utils.logger import logger
import os

class Settings:

    DQL: str = "dql"
    QL: str = "ql"

    load: bool
    train: bool
    visual: bool
    session: int
    load_file: Optional[str]
    train_file: Optional[str]
    agent_type: str

    _args: Namespace

    def __init__(self):
        self.load = False
        self.visual = False
        self.train = False
        self.session = 1
        self.load_file = None
        self.train_file = None
        self._args = None

    def parse(self):
        if self._args is None:
            self._parse()

    def _parse(self):
        parser = ArgumentParser(
            prog="snake",
            description="Reinforcement Learning project implementing a snake game." 
        )
        parser.add_argument("--visual", action="store_true", default=False)
        parser.add_argument("--train", type=str)
        parser.add_argument("--load", type=str)
        parser.add_argument("--session", type=int, default=1)
        parser.add_argument("--agent", type=str, choices=['dql', 'ql'], default='ql')
        self._args = parser.parse_args()
        self._parse_visual()
        self._parse_train()
        self._parse_load()
        self._parse_session()
        self._parse_agent()


    def _parse_visual(self):
        self.visual = self._args.visual

    def _parse_train(self):
        self.train_file = self._args.train
        if self.train_file is not None:
            self.train_file = self.train_file.strip()
            if not self.train_file:
                logger.error("Bad input on train argument")
                os._exit(1)
            self.train = True
        else:
            self.train = False

    def _parse_load(self):
        self.load_file = self._args.load
        if self.load_file is not None:
            self.load_file = self.load_file.strip()
            if not self.load_file:
                logger.error("Bad input on load argument")
                os._exit(1)
            self.load = True
        else:
            self.load = False

    def _parse_session(self):
        self.session = self._args.session
        if self.session <= 0:
            logger.error("Session should be striclty superior to 0")
            os._exit(1)

    def _parse_agent(self):
        self.agent_type = self._args.agent