from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.core.singletons import initialize_singletons, shutdown_singletons
from app.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize singletons on startup, shutdown on exit."""
    initialize_singletons()
    try:
        yield
    finally:
        shutdown_singletons()


app = FastAPI(title="Proposal Intelligence API", lifespan=lifespan)

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main:app --reload
