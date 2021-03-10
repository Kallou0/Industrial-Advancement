"""Initial migration.

Revision ID: c1c3a4095d80
Revises: 
Create Date: 2021-03-09 08:05:54.487212

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1c3a4095d80'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Industry',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('year', sa.String(length=150), nullable=False),
    sa.Column('agric', sa.String(length=564), nullable=True),
    sa.Column('mining', sa.String(length=564), nullable=True),
    sa.Column('manufacturing', sa.String(length=564), nullable=True),
    sa.Column('electricity_water', sa.String(length=564), nullable=True),
    sa.Column('construction', sa.String(length=564), nullable=True),
    sa.Column('distribution', sa.String(length=564), nullable=True),
    sa.Column('transport', sa.String(length=564), nullable=True),
    sa.Column('financial', sa.String(length=564), nullable=True),
    sa.Column('real_estate', sa.String(length=564), nullable=True),
    sa.Column('public_administration', sa.String(length=564), nullable=True),
    sa.Column('education', sa.String(length=564), nullable=True),
    sa.Column('human_health', sa.String(length=564), nullable=True),
    sa.Column('domestic_services', sa.String(length=564), nullable=True),
    sa.Column('net_tax', sa.String(length=564), nullable=True),
    sa.Column('GDP', sa.String(length=564), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('year')
    )
    op.create_table('NetExports',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('year', sa.String(length=150), nullable=False),
    sa.Column('goods_imports', sa.String(length=564), nullable=True),
    sa.Column('goods_exports', sa.String(length=564), nullable=True),
    sa.Column('services_exports', sa.String(length=564), nullable=True),
    sa.Column('services_imports', sa.String(length=564), nullable=True),
    sa.Column('bop_goods', sa.String(length=564), nullable=True),
    sa.Column('bop_services', sa.String(length=564), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('year')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('NetExports')
    op.drop_table('Industry')
    # ### end Alembic commands ###
