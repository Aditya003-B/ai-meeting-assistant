"""Custom exception classes for the application."""

from typing import Any


class AppException(Exception):
    """Base exception class for all application exceptions."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize exception with message and optional details."""
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class TranscriptionError(AppException):
    """Raised when audio transcription fails."""

    pass


class EmbeddingError(AppException):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(AppException):
    """Raised when vector store operations fail."""

    pass


class ExtractionError(AppException):
    """Raised when LangGraph extraction agent fails."""

    pass


class RAGError(AppException):
    """Raised when RAG chain fails."""

    pass


class ValidationError(AppException):
    """Raised when data validation fails."""

    pass


class DatabaseError(AppException):
    """Raised when database operations fail."""

    pass


class FileUploadError(AppException):
    """Raised when file upload validation fails."""

    pass


class MeetingNotFoundError(AppException):
    """Raised when a meeting is not found in the database."""

    pass


class InvalidAudioFormatError(FileUploadError):
    """Raised when uploaded audio file has invalid format."""

    pass


class AudioFileTooLargeError(FileUploadError):
    """Raised when uploaded audio file exceeds size limit."""

    pass
