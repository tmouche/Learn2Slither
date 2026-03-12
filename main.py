from fastapi import FastAPI
from typing import AsyncGenerator

async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    
    try:
        yield
    finally:
        pass 

app = FastAPI(lifespan=lifespan)