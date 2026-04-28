"""STT (Speech-to-Text) API routes."""

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status

from app.api.dependencies import get_current_user
from app.api.schemas.stt import SttRequest, SttResponse
from app.services.stt import SttSaturatedError, transcribe, transcribe_file

router = APIRouter()
logger = logging.getLogger(__name__)

SATURATED_RETRY_AFTER_SECONDS = 30
UPLOAD_CHUNK_SIZE = 64 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".webm", ".mp3", ".wav", ".m4a", ".ogg", ".flac"}


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


@router.post("/stt/upload", response_model=SttResponse)
async def speech_to_text_upload(
    response: Response,
    file: UploadFile = File(..., description="Audio file (webm, mp3, wav, m4a, ogg, flac)"),
    language: str = Form("ko", description="Language code hint (e.g., 'ko', 'en')"),
    prompt: str | None = Form(None, description="Optional initial prompt for biasing"),
    current_user: dict = Depends(get_current_user),
):
    """Multipart upload variant. Streams uploaded audio to a temp file, then transcribes."""
    suffix = os.path.splitext(file.filename or "audio.webm")[1].lower() or ".webm"
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio extension: {suffix}",
        )

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)

        try:
            result = await transcribe_file(audio_path=temp_path, language=language, prompt=prompt)
        except SttSaturatedError as exc:
            response.headers["Retry-After"] = str(SATURATED_RETRY_AFTER_SECONDS)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="STT worker saturated. Try again later.",
            ) from exc
        except Exception as exc:
            logger.exception("STT upload processing failed for filename=%s", file.filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="STT processing failed due to an internal error.",
            ) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return SttResponse(**result)
