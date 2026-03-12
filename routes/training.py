from fastapi import APIRouter, status
from services.agent_ql import AgentQL
from services.environment import Environment
from utils.dataclass import TrainingNew
training_router = APIRouter(prefix="training")


@training_router.post("/new/dqn", status_code=status.HTTP_202_ACCEPTED)
async def new_dqn(param: DqnNew):
    pass

@training_router.post("/new/dql", status_code=status.HTTP_202_ACCEPTED)
async def new_dql(param: DqlNew):
    pass

@training_router.post("/new/ql", status_code=status.HTTP_202_ACCEPTED)
async def new_ql(params: TrainingNew):
    env = Environment(params.env_params)
    agent = AgentQL(env, params.agent_params)
    return 

