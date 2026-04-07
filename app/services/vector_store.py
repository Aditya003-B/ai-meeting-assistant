"""Vector store operations for pgvector."""

import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import VectorStoreError
from app.models.db import Meeting
from app.services.embeddings import embedding_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """Service for vector store operations using pgvector."""

    async def index_meeting(
        self,
        db: AsyncSession,
        meeting_id: uuid.UUID,
        content: str,
    ) -> None:
        """Generate and store embedding for a meeting."""
        try:
            logger.info(f"Indexing meeting {meeting_id}")

            embedding = await embedding_service.generate_embedding(content)

            stmt = select(Meeting).where(Meeting.id == meeting_id)
            result = await db.execute(stmt)
            meeting = result.scalar_one_or_none()

            if not meeting:
                raise VectorStoreError(
                    message=f"Meeting {meeting_id} not found",
                    details={"meeting_id": str(meeting_id)},
                )

            meeting.embedding = embedding
            await db.commit()

            logger.info(f"Successfully indexed meeting {meeting_id}")

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Failed to index meeting {meeting_id}: {str(e)}")
            raise VectorStoreError(
                message="Failed to index meeting",
                details={"meeting_id": str(meeting_id), "error": str(e)},
            ) from e

    async def similarity_search(
        self,
        db: AsyncSession,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[Meeting, float]]:
        """Perform similarity search using cosine distance."""
        try:
            logger.info(f"Performing similarity search with top_k={top_k}")

            query_embedding = await embedding_service.generate_embedding(query_text)

            query = text(
                """
                SELECT 
                    id, title, transcript, summary, action_items, decisions,
                    confidence, status, created_at, updated_at,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM meetings
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
                """
            )

            result = await db.execute(
                query,
                {"embedding": str(query_embedding), "limit": top_k},
            )

            rows = result.fetchall()

            results: list[tuple[Meeting, float]] = []
            for row in rows:
                meeting = Meeting(
                    id=row.id,
                    title=row.title,
                    transcript=row.transcript,
                    summary=row.summary,
                    action_items=row.action_items,
                    decisions=row.decisions,
                    confidence=row.confidence,
                    status=row.status,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                similarity_score = float(row.similarity)
                results.append((meeting, similarity_score))

            logger.info(f"Found {len(results)} similar meetings")

            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise VectorStoreError(
                message="Failed to perform similarity search",
                details={"error": str(e), "top_k": top_k},
            ) from e

    async def reindex_all_meetings(self, db: AsyncSession) -> int:
        """Reindex all meetings that don't have embeddings."""
        try:
            logger.info("Starting reindexing of all meetings")

            stmt = select(Meeting).where(Meeting.embedding.is_(None))
            result = await db.execute(stmt)
            meetings = result.scalars().all()

            count = 0
            for meeting in meetings:
                content = f"{meeting.title}\n\n{meeting.summary or ''}\n\n{meeting.transcript}"
                embedding = await embedding_service.generate_embedding(content)
                meeting.embedding = embedding
                count += 1

            await db.commit()

            logger.info(f"Reindexed {count} meetings")

            return count

        except Exception as e:
            logger.error(f"Reindexing failed: {str(e)}")
            raise VectorStoreError(
                message="Failed to reindex meetings",
                details={"error": str(e)},
            ) from e


vector_store_service = VectorStoreService()
