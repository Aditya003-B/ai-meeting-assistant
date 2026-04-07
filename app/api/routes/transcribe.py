"""Transcription API routes."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.exceptions import AudioFileTooLargeError, InvalidAudioFormatError
from app.models.schemas import ErrorResponse, TranscriptionResponse
from app.services.transcription import transcription_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/transcribe", tags=["transcription"])

ALLOWED_AUDIO_FORMATS = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg"}


async def validate_audio_file(file: UploadFile) -> None:
    """Validate audio file format and size."""
    if not file.filename:
        raise InvalidAudioFormatError(
            message="Filename is required",
            details={"filename": None},
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_AUDIO_FORMATS:
        raise InvalidAudioFormatError(
            message=f"Invalid audio format. Allowed formats: {', '.join(ALLOWED_AUDIO_FORMATS)}",
            details={"filename": file.filename, "extension": file_ext},
        )

    content = await file.read()
    await file.seek(0)

    if len(content) > settings.max_audio_size_bytes:
        raise AudioFileTooLargeError(
            message=f"Audio file exceeds maximum size of {settings.max_audio_size_mb}MB",
            details={
                "filename": file.filename,
                "size_bytes": len(content),
                "max_size_bytes": settings.max_audio_size_bytes,
            },
        )


@router.post(
    "",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe audio file",
    description="Upload an audio file and get back the transcribed text using OpenAI Whisper",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or size"},
        500: {"model": ErrorResponse, "description": "Transcription failed"},
    },
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
) -> TranscriptionResponse:
    """Transcribe an audio file using OpenAI Whisper API."""
    try:
        logger.info(f"Received transcription request for file: {file.filename}")

        await validate_audio_file(file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(file.filename or "audio.mp3").suffix,
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = Path(temp_file.name)

        try:
            result = await transcription_service.transcribe_audio(temp_file_path)

            return TranscriptionResponse(
                transcript=result["transcript"],
                duration_seconds=result["duration_seconds"],
                language=result["language"],
            )

        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    except (InvalidAudioFormatError, AudioFileTooLargeError) as e:
        logger.warning(f"File validation failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.__class__.__name__, "message": e.message, "details": e.details},
        ) from e

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "TranscriptionError",
                "message": "Failed to transcribe audio file",
                "details": {"error": str(e)},
            },
        ) from e
