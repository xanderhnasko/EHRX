"""
Lightweight Postgres client utilities.

Uses psycopg (v3) with a small schema for documents and extractions.
"""

import psycopg

from ehrx.db.config import DBConfig


SCHEMA_SQL = """
create extension if not exists "uuid-ossp";

create table if not exists documents (
  id uuid primary key default uuid_generate_v4(),
  original_filename text not null,
  storage_url text not null,
  sha256 text,
  created_at timestamptz not null default now()
);

create table if not exists extractions (
  id uuid primary key default uuid_generate_v4(),
  document_id uuid not null references documents(id) on delete cascade,
  kind text not null check (kind in ('full','enhanced','index')),
  storage_url text not null,
  total_pages int,
  total_elements int,
  metadata jsonb,
  created_at timestamptz not null default now(),
  unique (document_id, kind)
);
"""


def get_conn(config: DBConfig) -> psycopg.Connection:
    """Open a blocking connection (use proxy or private IP)."""
    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.name,
        user=config.user,
        password=config.password,
        sslmode=config.sslmode,
    )


def init_schema(conn: psycopg.Connection) -> None:
    """Create tables/extensions if missing."""
    with conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
