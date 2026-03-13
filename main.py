from fastapi import FastAPI
from typing import AsyncGenerator
from utils.agent_container import AgentContainer
import asyncio

async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    app.state.monitor = AgentContainer()

    asyncio.run(app.state.monitor.agent_train())

    try:
        yield
    finally:
        pass 

app = FastAPI(lifespan=lifespan)