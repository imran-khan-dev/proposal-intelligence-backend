"""Add workspace_id to documents and document_chunks for workspace isolation.

Revision ID: 002
Revises: 001
Create Date: 2026-11-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column(
            'workspace_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
            index=True,
        ),
    )
    op.add_column(
        'document_chunks',
        sa.Column(
            'workspace_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
            index=True,
        ),
    )


def downgrade() -> None:
    op.drop_column('document_chunks', 'workspace_id')
    op.drop_column('documents', 'workspace_id')
