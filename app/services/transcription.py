"""Audio transcription service using OpenAI Whisper API."""

from pathlib import Path

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import TranscriptionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptionService:
    """Service for transcribing audio files using OpenAI Whisper."""

    def __init__(self) -> None:
        """Initialize the transcription service with OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe_audio(
        self,
        audio_file_path: Path,
        language: str | None = None,
    ) -> dict[str, str | float]:
        """Transcribe audio file using Whisper API."""
        try:
            logger.info(f"Starting transcription for file: {audio_file_path}")

            with open(audio_file_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                )

            result = {
                "transcript": response.text,
                "duration_seconds": response.duration or 0.0,
                "language": response.language or "unknown",
            }

            logger.info(
                f"Transcription completed: {len(result['transcript'])} chars, "
                f"{result['duration_seconds']}s, language={result['language']}"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionError(
                message="Failed to transcribe audio file",
                details={"error": str(e), "file": str(audio_file_path)},
            ) from e


transcription_service = TranscriptionService()
