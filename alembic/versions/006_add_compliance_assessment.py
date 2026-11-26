"""Add compliance assessment columns to draft generation runs.

Revision ID: 006
Revises: 005
Create Date: 2026-11-26
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable so existing draft rows (without an assessment) remain readable;
    # the API treats a NULL status as ``needs_review``.
    op.add_column(
        'draft_generation_runs',
        sa.Column('compliance_status', sa.String(length=32), nullable=True),
    )
    op.add_column(
        'draft_generation_runs',
        sa.Column('compliance_rationale', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('draft_generation_runs', 'compliance_rationale')
    op.drop_column('draft_generation_runs', 'compliance_status')