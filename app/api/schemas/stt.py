"""STT (Speech-to-Text) related schemas."""

from pydantic import BaseModel, Field


class SttSegment(BaseModel):
    start_ms: int = Field(..., description="Segment start (milliseconds)")
    end_ms: int = Field(..., description="Segment end (milliseconds)")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: float = Field(..., description="Average log probability (model confidence)")


class SttRequest(BaseModel):
    audio_url: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pre-signed S3 URL of the audio file (mp3, wav, m4a, etc.)",
    )
    language: str = Field(
        default="ko",
        description="Language code hint for transcription (e.g., 'ko', 'en')",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional initial prompt that biases the transcription",
    )


class SttResponse(BaseModel):
    language: str = Field(..., description="Detected/used language code")
    duration_ms: int = Field(..., description="Audio duration (milliseconds)")
    text: str = Field(..., description="Concatenated transcript text")
    segments: list[SttSegment] = Field(..., description="Per-segment transcription details")
