
from fastapi import APIRouter, File, UploadFile

from app.modules.rfp_analyze.controller import analyze_rfp_controller


router = APIRouter(tags=["rfp"])


@router.post("/analyze")
async def analyze_rfp_route(
    file: UploadFile = File(...),
):
    return await analyze_rfp_controller(file)
