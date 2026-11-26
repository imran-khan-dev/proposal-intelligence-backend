"""Add source_type to documents to distinguish RFP vs company knowledge.

Revision ID: 005
Revises: 004
Create Date: 2026-11-26
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ``source_type`` distinguishes document categories that share the same
    # workspace-scoped vector store.  Existing rows default to ``'rfp'`` so
    # that legacy documents are never accidentally treated as company
    # knowledge by requirement drafting.
    op.add_column(
        'documents',
        sa.Column(
            'source_type',
            sa.String(length=32),
            nullable=False,
            server_default='rfp',
        ),
    )
    op.create_index(
        'ix_documents_source_type',
        'documents',
        ['source_type'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_documents_source_type', table_name='documents')
    op.drop_column('documents', 'source_type')