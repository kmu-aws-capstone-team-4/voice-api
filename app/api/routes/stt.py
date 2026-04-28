"""STT (Speech-to-Text) API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import get_current_user
from app.api.schemas.stt import SttRequest, SttResponse
from app.services.stt import SttSaturatedError, transcribe

router = APIRouter()
logger = logging.getLogger(__name__)

SATURATED_RETRY_AFTER_SECONDS = 30


@router.post("/stt", response_model=SttResponse)
async def speech_to_text(
    request: SttRequest,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    """Transcribe a remote audio file using faster-whisper. Returns 503 + Retry-After when saturated."""
    try:
        result = await transcribe(
            audio_url=request.audio_url,
            language=request.language,
            prompt=request.prompt,
        )
    except SttSaturatedError as exc:
        response.headers["Retry-After"] = str(SATURATED_RETRY_AFTER_SECONDS)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT worker saturated. Try again later.",
        ) from exc
    except Exception as exc:
        logger.exception("STT processing failed for audio_url=%s", request.audio_url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="STT processing failed due to an internal error.",
        ) from exc

    return SttResponse(**result)
