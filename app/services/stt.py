"""faster-whisper based STT service (CPU mode, semaphore queue, model singleton)."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager

import aiohttp
from faster_whisper import WhisperModel

_MODEL_LOCK = asyncio.Lock()
_INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
_MODEL: WhisperModel | None = None

DEFAULT_MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "small")
DEFAULT_DEVICE = os.environ.get("STT_DEVICE", "cpu")
DEFAULT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")
AUDIO_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("STT_AUDIO_DOWNLOAD_TIMEOUT", "60"))


class SttSaturatedError(Exception):
    """Raised when the inference semaphore is full and the request should be retried later."""


async def get_model() -> WhisperModel:
    """Lazy singleton accessor for the WhisperModel instance."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    async with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = WhisperModel(
                DEFAULT_MODEL_SIZE,
                device=DEFAULT_DEVICE,
                compute_type=DEFAULT_COMPUTE_TYPE,
            )
    return _MODEL


@asynccontextmanager
async def acquire_inference_slot():
    """Acquire the inference semaphore or raise SttSaturatedError immediately if full."""
    acquired = False
    try:
        acquired = await asyncio.wait_for(_INFERENCE_SEMAPHORE.acquire(), timeout=0.05)
    except TimeoutError as exc:
        raise SttSaturatedError("STT worker saturated") from exc

    try:
        yield
    finally:
        if acquired:
            _INFERENCE_SEMAPHORE.release()


async def download_audio(audio_url: str) -> str:
    """Download the remote audio file into a temporary path and return its filename."""
    suffix = ".mp3"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    timeout = aiohttp.ClientTimeout(total=AUDIO_DOWNLOAD_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session, session.get(audio_url) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as out:
            async for chunk in response.content.iter_chunked(64 * 1024):
                out.write(chunk)

    return temp_path


def _run_transcription(model: WhisperModel, path: str, language: str, prompt: str | None) -> dict:
    """Synchronous faster-whisper transcribe call wrapped to run in a thread."""
    segments_iter, info = model.transcribe(
        path,
        language=language,
        initial_prompt=prompt,
        beam_size=1,
        vad_filter=True,
    )

    segments = []
    text_parts = []
    for segment in segments_iter:
        text_parts.append(segment.text)
        segments.append(
            {
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "text": segment.text,
                "confidence": float(getattr(segment, "avg_logprob", 0.0)),
            }
        )

    return {
        "language": info.language or language,
        "duration_ms": int(info.duration * 1000),
        "text": "".join(text_parts).strip(),
        "segments": segments,
    }


async def transcribe(audio_url: str, language: str, prompt: str | None = None) -> dict:
    """Download audio_url, run faster-whisper inference under the semaphore, and return results."""
    audio_path = await download_audio(audio_url)
    try:
        async with acquire_inference_slot():
            model = await get_model()
            return await asyncio.to_thread(_run_transcription, model, audio_path, language, prompt)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
