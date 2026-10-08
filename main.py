from fastapi import FastAPI

from app.api.rfp import router as rfp_router

app = FastAPI(title="Proposal Intelligence API")

app.include_router(rfp_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main:app --reload
# alembic revision --autogenerate -m "add extracted text to documents"
# alembic upgrade head