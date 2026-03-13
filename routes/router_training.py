from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status
)
from services.agent import Agent
from services.agent_ql import AgentQL
from services.environment import Environment
from utils.dataclass import TrainingNew
from utils.logger import Logger

logger = Logger()

router_training = APIRouter(prefix="training")

@router_training.post("/new/dqn", status_code=status.HTTP_202_ACCEPTED)
async def new_dqn(param: DqnNew):
    pass

@router_training.post("/new/dql", status_code=status.HTTP_202_ACCEPTED)
async def new_dql(param: DqlNew):
    pass

@router_training.post("/new/ql", status_code=status.HTTP_202_ACCEPTED)
async def new_ql(
    request: Request,
    params: TrainingNew
):
    env = Environment(params.env_params)
    agent = AgentQL(env, params.agent_params, name=params.agent_params.agent_name)
    request.app.state.monitor.add_agent(agent)
    return agent.name

@router_training.get("/info/agent/state", status_code=status.HTTP_200_OK)
async def get_state(
    request: Request,
    agent_name: str
):
    try:
        agent: Agent = request.app.state.monitor.get_agent(agent_name)
    except Exception as exc:
        logger.error(str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return agent.state

@router_training.get("/info/agent/map", status_code=status.HTTP_200_OK)
async def get_map(
    request: Request,
    agent_name: str
):
    try:
        agent: Agent = request.app.state.monitor.get_agent(agent_name)
    except Exception as exc:
        logger.error(str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return agent.get_map()


