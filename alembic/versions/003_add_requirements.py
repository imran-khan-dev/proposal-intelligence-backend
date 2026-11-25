"""Add extraction_runs and requirements tables for structured requirement extraction.

Revision ID: 003
Revises: 002
Create Date: 2026-11-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extraction run: one requirement-extraction pass over an ingested doc.
    op.create_table(
        'extraction_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('doc_id', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('requirement_count', sa.Integer(), nullable=False),
        sa.Column('errors', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_extraction_runs_workspace_id', 'extraction_runs', ['workspace_id'], unique=False)
    op.create_index('ix_extraction_runs_document_id', 'extraction_runs', ['document_id'], unique=False)
    op.create_index('ix_extraction_runs_doc_id', 'extraction_runs', ['doc_id'], unique=False)

    # Extracted requirement: individually identifiable, traceable requirement.
    op.create_table(
        'requirements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('requirement_key', sa.String(length=255), nullable=False),
        sa.Column('doc_id', sa.String(length=255), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('section_title', sa.String(length=500), nullable=True),
        sa.Column('section_path', sa.String(length=1000), nullable=True),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('character_start', sa.Integer(), nullable=False),
        sa.Column('character_end', sa.Integer(), nullable=False),
        sa.Column('modality', sa.String(length=32), nullable=False),
        sa.Column('category', sa.String(length=64), nullable=False),
        sa.Column('cross_references', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('uncertainty_notes', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('extraction_method', sa.String(length=16), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['run_id'], ['extraction_runs.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('run_id', 'requirement_key', name='uq_requirements_run_key'),
    )
    op.create_index('ix_requirements_workspace_id', 'requirements', ['workspace_id'], unique=False)
    op.create_index('ix_requirements_document_id', 'requirements', ['document_id'], unique=False)
    op.create_index('ix_requirements_run_id', 'requirements', ['run_id'], unique=False)
    op.create_index('ix_requirements_doc_id', 'requirements', ['doc_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_requirements_doc_id', table_name='requirements')
    op.drop_index('ix_requirements_run_id', table_name='requirements')
    op.drop_index('ix_requirements_document_id', table_name='requirements')
    op.drop_index('ix_requirements_workspace_id', table_name='requirements')
    op.drop_table('requirements')

    op.drop_index('ix_extraction_runs_doc_id', table_name='extraction_runs')
    op.drop_index('ix_extraction_runs_document_id', table_name='extraction_runs')
    op.drop_index('ix_extraction_runs_workspace_id', table_name='extraction_runs')
    op.drop_table('extraction_runs')