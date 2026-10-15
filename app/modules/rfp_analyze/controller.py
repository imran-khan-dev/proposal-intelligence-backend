import os
import tempfile

from fastapi import HTTPException, UploadFile

from app.modules.rfp_analyze.schema import RFPAnalysis
from app.modules.rfp_analyze.service import analyze_rfp

ALLOWED_CONTENT_TYPES = {"application/pdf"}


async def analyze_rfp_controller(file: UploadFile) -> RFPAnalysis:
    """Validate the upload and coordinate RFP analysis."""

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. "
                f"Expected PDF, got {file.content_type}."
            ),
        )

    temp_path = None

    try:
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
                status_code=400,
                detail=f"Failed to read uploaded file: {exc}",
            ) from exc

        try:
            return analyze_rfp(temp_path)
        except Exception as exc:
            from app.ai.error_handling import handle_ai_exception
            raise handle_ai_exception(exc)

    finally:
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
