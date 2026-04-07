"""Pydantic schemas for request/response validation."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """Schema for an action item."""

    owner: str = Field(..., description="Person responsible for the action")
    task: str = Field(..., description="Description of the task")
    deadline: str | None = Field(None, description="Deadline if mentioned")


class Decision(BaseModel):
    """Schema for a decision made in the meeting."""

    decision: str = Field(..., description="The decision that was made")
    context: str | None = Field(None, description="Context around the decision")


class TranscriptionResponse(BaseModel):
    """Response schema for transcription endpoint."""

    transcript: str = Field(..., description="Transcribed text from audio")
    duration_seconds: float = Field(..., description="Duration of audio in seconds")
    language: str = Field(..., description="Detected language of the audio")


class MeetingCreate(BaseModel):
    """Request schema for creating a meeting."""

    title: str = Field(..., min_length=1, max_length=255, description="Meeting title")
    transcript: str = Field(..., min_length=1, description="Meeting transcript")


class MeetingConfirm(BaseModel):
    """Request schema for confirming/correcting meeting extraction."""

    summary: str = Field(..., description="Confirmed or corrected summary")
    action_items: list[ActionItem] = Field(default_factory=list, description="Action items")
    decisions: list[Decision] = Field(default_factory=list, description="Decisions made")


class MeetingResponse(BaseModel):
    """Response schema for meeting endpoints."""

    id: uuid.UUID = Field(..., description="Meeting ID")
    title: str = Field(..., description="Meeting title")
    transcript: str = Field(..., description="Meeting transcript")
    summary: str | None = Field(None, description="Meeting summary")
    action_items: list[dict] = Field(default_factory=list, description="Extracted action items")
    decisions: list[dict] = Field(default_factory=list, description="Decisions made")
    confidence: float | None = Field(None, description="Extraction confidence score")
    status: str = Field(..., description="Meeting status: pending_review or confirmed")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        """Pydantic config."""

        from_attributes = True


class MeetingListResponse(BaseModel):
    """Response schema for paginated meeting list."""

    meetings: list[MeetingResponse] = Field(..., description="List of meetings")
    total: int = Field(..., description="Total number of meetings")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")


class QueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""

    question: str = Field(..., min_length=1, description="Question to ask about meetings")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")


class QuerySource(BaseModel):
    """Schema for a source document in query response."""

    meeting_id: uuid.UUID = Field(..., description="ID of the source meeting")
    title: str = Field(..., description="Title of the source meeting")
    relevance_score: float = Field(..., description="Relevance score of the source")


class QueryResponse(BaseModel):
    """Response schema for RAG query endpoint."""

    answer: str = Field(..., description="Generated answer to the question")
    sources: list[QuerySource] = Field(..., description="Source meetings used for the answer")


class EvalResult(BaseModel):
    """Response schema for evaluation endpoint."""

    rag_accuracy: float = Field(..., description="RAG retrieval accuracy score")
    extraction_f1: float = Field(..., description="Extraction F1 score")
    regression_delta: float = Field(..., description="Regression delta vs previous run")


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")
