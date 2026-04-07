"""Initial migration: create meetings table with pgvector

Revision ID: 001_initial
Revises: 
Create Date: 2026-04-07 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    op.create_table(
        'meetings',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('transcript', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('action_items', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column('decisions', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=50), server_default=sa.text("'confirmed'"), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index(op.f('ix_meetings_title'), 'meetings', ['title'], unique=False)
    op.create_index(op.f('ix_meetings_status'), 'meetings', ['status'], unique=False)
    
    op.execute(
        "CREATE INDEX meetings_embedding_idx ON meetings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.drop_index('meetings_embedding_idx', table_name='meetings')
    op.drop_index(op.f('ix_meetings_status'), table_name='meetings')
    op.drop_index(op.f('ix_meetings_title'), table_name='meetings')
    op.drop_table('meetings')
    op.execute('DROP EXTENSION IF EXISTS vector')
