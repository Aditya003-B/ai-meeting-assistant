"""Meeting management API routes."""

import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.extractor import extraction_agent
from app.core.database import get_db
from app.core.exceptions import MeetingNotFoundError
from app.models.db import Meeting
from app.models.schemas import (
    ErrorResponse,
    MeetingConfirm,
    MeetingCreate,
    MeetingListResponse,
    MeetingResponse,
)
from app.services.vector_store import vector_store_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post(
    "",
    response_model=MeetingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new meeting",
    description="Create a meeting from transcript, extract summary and action items using LangGraph agent",
    responses={
        500: {"model": ErrorResponse, "description": "Extraction or database error"},
    },
)
async def create_meeting(
    meeting_data: MeetingCreate,
    db: AsyncSession = Depends(get_db),
) -> MeetingResponse:
    """Create a new meeting and extract information using LangGraph agent."""
    try:
        logger.info(f"Creating meeting: {meeting_data.title}")

        extraction_result = await extraction_agent.extract(meeting_data.transcript)

        meeting = Meeting(
            title=meeting_data.title,
            transcript=meeting_data.transcript,
            summary=extraction_result["summary"],
            action_items=extraction_result["action_items"],
            decisions=extraction_result["decisions"],
            confidence=extraction_result["confidence"],
            status="pending_review" if extraction_result["requires_human_review"] else "confirmed",
        )

        db.add(meeting)
        await db.commit()
        await db.refresh(meeting)

        if not extraction_result["requires_human_review"]:
            content = f"{meeting.title}\n\n{meeting.summary}\n\n{meeting.transcript}"
            await vector_store_service.index_meeting(
                db=db,
                meeting_id=meeting.id,
                content=content,
            )

        logger.info(
            f"Meeting created: {meeting.id}, status={meeting.status}, "
            f"confidence={meeting.confidence:.2f}"
        )

        return MeetingResponse.model_validate(meeting)

    except Exception as e:
        logger.error(f"Failed to create meeting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MeetingCreationError",
                "message": "Failed to create meeting",
                "details": {"error": str(e)},
            },
        ) from e


@router.patch(
    "/{meeting_id}/confirm",
    response_model=MeetingResponse,
    status_code=status.HTTP_200_OK,
    summary="Confirm and correct meeting extraction",
    description="Human-in-the-loop endpoint to confirm or correct extracted information",
    responses={
        404: {"model": ErrorResponse, "description": "Meeting not found"},
        500: {"model": ErrorResponse, "description": "Update failed"},
    },
)
async def confirm_meeting(
    meeting_id: uuid.UUID,
    confirmation: MeetingConfirm,
    db: AsyncSession = Depends(get_db),
) -> MeetingResponse:
    """Confirm or correct meeting extraction results."""
    try:
        logger.info(f"Confirming meeting: {meeting_id}")

        stmt = select(Meeting).where(Meeting.id == meeting_id)
        result = await db.execute(stmt)
        meeting = result.scalar_one_or_none()

        if not meeting:
            raise MeetingNotFoundError(
                message=f"Meeting {meeting_id} not found",
                details={"meeting_id": str(meeting_id)},
            )

        meeting.summary = confirmation.summary
        meeting.action_items = [item.model_dump() for item in confirmation.action_items]
        meeting.decisions = [decision.model_dump() for decision in confirmation.decisions]
        meeting.status = "confirmed"

        await db.commit()
        await db.refresh(meeting)

        content = f"{meeting.title}\n\n{meeting.summary}\n\n{meeting.transcript}"
        await vector_store_service.index_meeting(
            db=db,
            meeting_id=meeting.id,
            content=content,
        )

        logger.info(f"Meeting confirmed: {meeting_id}")

        return MeetingResponse.model_validate(meeting)

    except MeetingNotFoundError as e:
        logger.warning(f"Meeting not found: {meeting_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": e.__class__.__name__, "message": e.message, "details": e.details},
        ) from e

    except Exception as e:
        logger.error(f"Failed to confirm meeting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MeetingUpdateError",
                "message": "Failed to confirm meeting",
                "details": {"error": str(e)},
            },
        ) from e


@router.get(
    "",
    response_model=MeetingListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all meetings",
    description="Get paginated list of meetings with optional search",
    responses={
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def list_meetings(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search keyword in title or summary"),
    db: AsyncSession = Depends(get_db),
) -> MeetingListResponse:
    """List meetings with pagination and optional search."""
    try:
        logger.info(f"Listing meetings: page={page}, limit={limit}, search={search}")

        offset = (page - 1) * limit

        count_stmt = select(func.count(Meeting.id))
        query_stmt = select(Meeting).order_by(Meeting.created_at.desc())

        if search:
            search_filter = (
                Meeting.title.ilike(f"%{search}%") | Meeting.summary.ilike(f"%{search}%")
            )
            count_stmt = count_stmt.where(search_filter)
            query_stmt = query_stmt.where(search_filter)

        count_result = await db.execute(count_stmt)
        total = count_result.scalar_one()

        query_stmt = query_stmt.offset(offset).limit(limit)
        result = await db.execute(query_stmt)
        meetings = result.scalars().all()

        pages = math.ceil(total / limit) if total > 0 else 0

        logger.info(f"Found {total} meetings, returning page {page}/{pages}")

        return MeetingListResponse(
            meetings=[MeetingResponse.model_validate(m) for m in meetings],
            total=total,
            page=page,
            limit=limit,
            pages=pages,
        )

    except Exception as e:
        logger.error(f"Failed to list meetings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DatabaseError",
                "message": "Failed to list meetings",
                "details": {"error": str(e)},
            },
        ) from e


@router.get(
    "/{meeting_id}",
    response_model=MeetingResponse,
    status_code=status.HTTP_200_OK,
    summary="Get meeting by ID",
    description="Retrieve full meeting details including transcript and extracted information",
    responses={
        404: {"model": ErrorResponse, "description": "Meeting not found"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def get_meeting(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> MeetingResponse:
    """Get a specific meeting by ID."""
    try:
        logger.info(f"Fetching meeting: {meeting_id}")

        stmt = select(Meeting).where(Meeting.id == meeting_id)
        result = await db.execute(stmt)
        meeting = result.scalar_one_or_none()

        if not meeting:
            raise MeetingNotFoundError(
                message=f"Meeting {meeting_id} not found",
                details={"meeting_id": str(meeting_id)},
            )

        return MeetingResponse.model_validate(meeting)

    except MeetingNotFoundError as e:
        logger.warning(f"Meeting not found: {meeting_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": e.__class__.__name__, "message": e.message, "details": e.details},
        ) from e

    except Exception as e:
        logger.error(f"Failed to get meeting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DatabaseError",
                "message": "Failed to get meeting",
                "details": {"error": str(e)},
            },
        ) from e
