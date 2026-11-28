"""Add review_audit_events table for append-only human review audit history (M4.2).

Revision ID: 008
Revises: 007
Create Date: 2026-11-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'review_audit_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('requirement_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('draft_run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('previous_review_status', sa.String(length=32), nullable=True),
        sa.Column('review_status', sa.String(length=32), nullable=False),
        sa.Column('decision', sa.String(length=32), nullable=False),
        sa.Column('reviewer_edited_answer', sa.Text(), nullable=True),
        sa.Column('reviewer_compliance_status', sa.String(length=32), nullable=True),
        sa.Column('reviewer_compliance_rationale', sa.Text(), nullable=True),
        sa.Column('reviewer_comment', sa.Text(), nullable=False),
        sa.Column('citation_reviews', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('reviewer_workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(['draft_run_id'], ['draft_generation_runs.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['requirement_id'], ['requirements.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_review_audit_events_workspace_id', 'review_audit_events', ['workspace_id'], unique=False)
    op.create_index('ix_review_audit_events_requirement_id', 'review_audit_events', ['requirement_id'], unique=False)
    op.create_index('ix_review_audit_events_draft_run_id', 'review_audit_events', ['draft_run_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_review_audit_events_draft_run_id', table_name='review_audit_events')
    op.drop_index('ix_review_audit_events_requirement_id', table_name='review_audit_events')
    op.drop_index('ix_review_audit_events_workspace_id', table_name='review_audit_events')
    op.drop_table('review_audit_events')