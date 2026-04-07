"""SQLAlchemy database models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Meeting(Base):
    """Meeting model with transcript, summary, action items, and vector embedding."""

    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_items: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    decisions: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'confirmed'"),
        index=True,
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        """String representation of Meeting."""
        return f"<Meeting(id={self.id}, title={self.title}, status={self.status})>"
