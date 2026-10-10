from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.rfp import router as rfp_router

app = FastAPI(title="Proposal Intelligence API")

app.include_router(rfp_router)
app.include_router(agent_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main:app --reload
