from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.router import router

app = FastAPI(title="Proposal Intelligence API")

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main:app --reload
