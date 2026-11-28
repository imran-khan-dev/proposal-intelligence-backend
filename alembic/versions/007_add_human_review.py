"""Add human review fields to draft generation runs.

Revision ID: 007
Revises: 006
Create Date: 2026-11-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Review lifecycle status: pending_review | approved | changes_requested
    op.add_column(
        'draft_generation_runs',
        sa.Column('review_status', sa.String(length=32), nullable=True),
    )
    # Human reviewer's edited answer (separate from AI-generated draft_answer)
    op.add_column(
        'draft_generation_runs',
        sa.Column('reviewer_edited_answer', sa.Text(), nullable=True),
    )
    # Human reviewer's compliance status override (separate from AI assessment)
    op.add_column(
        'draft_generation_runs',
        sa.Column('reviewer_compliance_status', sa.String(length=32), nullable=True),
    )
    # Human reviewer's compliance rationale
    op.add_column(
        'draft_generation_runs',
        sa.Column('reviewer_compliance_rationale', sa.Text(), nullable=True),
    )
    # Reviewer comment/reason for the decision
    op.add_column(
        'draft_generation_runs',
        sa.Column('reviewer_comment', sa.Text(), nullable=True),
    )
    # Per-citation review decisions (distinct from automated M3.3 verification)
    # JSONB array of: {evidence_id, decision: "accept|reject|modify", reviewer_excerpt?, reason?}
    op.add_column(
        'draft_generation_runs',
        sa.Column('citation_reviews', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
    )
    # Timestamp when review was completed
    op.add_column(
        'draft_generation_runs',
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
    )
    # Identity/audit fields (feasible with current architecture - server-resolved workspace)
    op.add_column(
        'draft_generation_runs',
        sa.Column('reviewer_workspace_id', postgresql.UUID(as_uuid=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('draft_generation_runs', 'reviewer_workspace_id')
    op.drop_column('draft_generation_runs', 'reviewed_at')
    op.drop_column('draft_generation_runs', 'citation_reviews')
    op.drop_column('draft_generation_runs', 'reviewer_comment')
    op.drop_column('draft_generation_runs', 'reviewer_compliance_rationale')
    op.drop_column('draft_generation_runs', 'reviewer_compliance_status')
    op.drop_column('draft_generation_runs', 'reviewer_edited_answer')
    op.drop_column('draft_generation_runs', 'review_status')