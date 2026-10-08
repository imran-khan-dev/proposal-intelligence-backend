import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas.rfp import RFPAnalysis
from app.services.rfp_analyzer import analyze_rfp

router = APIRouter(prefix="/api/rfp", tags=["rfp"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}


@router.post("/analyze", response_model=RFPAnalysis)

async def analyze_rfp_endpoint(file: UploadFile = File(...)):

    """Accept an uploaded RFP PDF and return a structured analysis."""

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Expected PDF, got {file.content_type}.",
        )

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read uploaded file: {exc}"
        ) from exc
    finally:
        await file.close()

    try:
        result = analyze_rfp(temp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"AI provider request failed: {exc}"
        ) from exc
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return result