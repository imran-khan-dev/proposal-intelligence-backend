from fastapi import APIRouter

from app.schemas.rfp import RFPAnalysis

router = APIRouter(prefix="/api/rfp", tags=["rfp"])


@router.post("/analyze", response_model=RFPAnalysis)
async def analyze_rfp():
    """Temporary placeholder endpoint for RFP analysis."""
    return RFPAnalysis()