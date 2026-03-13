from collections import deque
from utils.exception import AgentNotFoundException
from utils.logger import Logger
from services.agent import Agent
from threading import Thread
from typing import Dict

import asyncio

logger = Logger()

class AgentContainer:

    in_training: bool = False

    loaded_agents: Dict[str, Agent]
    unloaded_agents: Dict[str, str]

    to_train: deque[Agent]

    active_thread: Thread | None = None

    def __init__(self):
        self.loaded_agents = {}
        self.unloaded_agents = {}
        self.__register_agents() # sert a faire un inventaire de tous les agents pas load 

    async def agent_train(self):
        while True:
            if self._is_thread_alive():
                continue
            try:
                next_agent: Agent = self.to_train.popleft()
            except IndexError:
                await asyncio.sleep(1)
                continue
            self.active_thread = Thread(target=next_agent.train, daemon=True)
            self.active_thread.start()

    async def _is_thread_alive(self):
        if self.active_thread:
            if self.active_thread.is_alive():
                await asyncio.sleep(1)
                return True
            else:
                self.active_thread = None
        return False

    def __register_agents(self):
        pass

    def add_agent(self, agent: Agent):
        self.loaded_agents[agent.name] = agent
        self.to_train.append(agent)

    def save_agent(self, agent: Agent):
        agent.save("path_to_file")

    def unload_agent(self, agent_name: str):
        self.loaded_agents.pop(agent_name, None)

    def delete_agent(self, agent_name: str):
        pass

    def load_agent(self, agent_name: str) -> Agent:
        pass

    def get_agent(self, agent_name: str) -> Agent:
        agent: Agent = self.loaded_agents.get(agent_name)
        if agent:
            return agent
        agent_path: str = self.unloaded_agents.get(agent_name)
        if not agent_path:
            logger.error(f"Agent requested {agent_name} not found")
            raise AgentNotFoundException()
        agent = self.load_agent(agent_name)
        if not agent:
            logger.critical("Agent in dict not know")
            self.unloaded_agents.pop(agent_name, None)
            raise AgentNotFoundException()
        return agent
        