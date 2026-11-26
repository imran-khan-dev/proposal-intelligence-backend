"""Add evidence_retrieval_runs, evidence_records, and draft_generation_runs tables for M3.

Revision ID: 004
Revises: 003
Create Date: 2026-11-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Evidence retrieval run: one evidence retrieval pass for a requirement.
    op.create_table(
        'evidence_retrieval_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('requirement_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
        sa.Column('evidence_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('errors', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['requirement_id'], ['requirements.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_evidence_retrieval_runs_workspace_id', 'evidence_retrieval_runs', ['workspace_id'], unique=False)
    op.create_index('ix_evidence_retrieval_runs_requirement_id', 'evidence_retrieval_runs', ['requirement_id'], unique=False)

    # Evidence record: a single piece of evidence retrieved for a requirement.
    op.create_table(
        'evidence_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_doc_id', sa.String(length=255), nullable=False),
        sa.Column('source_filename', sa.String(length=500), nullable=True),
        sa.Column('chunk_id', sa.String(length=255), nullable=True),
        sa.Column('chunk_index', sa.Integer(), nullable=True),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('character_start', sa.Integer(), nullable=True),
        sa.Column('character_end', sa.Integer(), nullable=True),
        sa.Column('section_path', sa.String(length=1000), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['run_id'], ['evidence_retrieval_runs.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_evidence_records_run_id', 'evidence_records', ['run_id'], unique=False)
    op.create_index('ix_evidence_records_workspace_id', 'evidence_records', ['workspace_id'], unique=False)
    op.create_index('ix_evidence_records_source_doc_id', 'evidence_records', ['source_doc_id'], unique=False)

    # Draft generation run: one draft generation pass for a requirement.
    op.create_table(
        'draft_generation_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('requirement_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('retrieval_run_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('draft_answer', sa.Text(), nullable=True),
        sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('evidence_found', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('uncertainty', sa.String(length=32), nullable=False, server_default='none'),
        sa.Column('uncertainty_notes', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
        sa.Column('errors', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['requirement_id'], ['requirements.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['retrieval_run_id'], ['evidence_retrieval_runs.id'], ondelete='SET NULL'),
    )
    op.create_index('ix_draft_generation_runs_workspace_id', 'draft_generation_runs', ['workspace_id'], unique=False)
    op.create_index('ix_draft_generation_runs_requirement_id', 'draft_generation_runs', ['requirement_id'], unique=False)
    op.create_index('ix_draft_generation_runs_retrieval_run_id', 'draft_generation_runs', ['retrieval_run_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_draft_generation_runs_retrieval_run_id', table_name='draft_generation_runs')
    op.drop_index('ix_draft_generation_runs_requirement_id', table_name='draft_generation_runs')
    op.drop_index('ix_draft_generation_runs_workspace_id', table_name='draft_generation_runs')
    op.drop_table('draft_generation_runs')

    op.drop_index('ix_evidence_records_source_doc_id', table_name='evidence_records')
    op.drop_index('ix_evidence_records_workspace_id', table_name='evidence_records')
    op.drop_index('ix_evidence_records_run_id', table_name='evidence_records')
    op.drop_table('evidence_records')

    op.drop_index('ix_evidence_retrieval_runs_requirement_id', table_name='evidence_retrieval_runs')
    op.drop_index('ix_evidence_retrieval_runs_workspace_id', table_name='evidence_retrieval_runs')
    op.drop_table('evidence_retrieval_runs')