from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.core.database import initialize_database, shutdown_database
from app.core.singletons import initialize_singletons, shutdown_singletons
from app.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize singletons and database on startup, shutdown on exit."""
    initialize_database()
    initialize_singletons()
    try:
        yield
    finally:
        shutdown_singletons()
        await shutdown_database()


app = FastAPI(title="Proposal Intelligence API", lifespan=lifespan)

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main:app --reload
