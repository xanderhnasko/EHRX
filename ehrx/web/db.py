"""
DB helpers for FastAPI service.
"""

import uuid
import json
from typing import Optional, List, Dict, Any

from psycopg.types.json import Jsonb

from ehrx.db.client import get_conn
from ehrx.db.config import DBConfig


class DB:
    def __init__(self, config: Optional[DBConfig] = None):
        self.config = config or DBConfig.from_env()

    def _conn(self):
        return get_conn(self.config)

    def create_document(self, original_filename: str, storage_url: str, sha256: Optional[str] = None) -> uuid.UUID:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                insert into documents (original_filename, storage_url, sha256)
                values (%s, %s, %s)
                returning id
                """,
                (original_filename, storage_url, sha256),
            )
            return cur.fetchone()[0]

    def get_document(self, document_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "select id, original_filename, storage_url, sha256, created_at from documents where id = %s",
                (document_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": str(row[0]),
                "original_filename": row[1],
                "storage_url": row[2],
                "sha256": row[3],
                "created_at": row[4].isoformat(),
            }

    def upsert_extraction(
        self,
        document_id: uuid.UUID,
        kind: str,
        storage_url: str,
        total_pages: Optional[int],
        total_elements: Optional[int],
        metadata: Optional[dict] = None,
    ) -> None:
        metadata_param = Jsonb(metadata) if metadata is not None else None
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                insert into extractions (document_id, kind, storage_url, total_pages, total_elements, metadata)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (document_id, kind) do update
                set storage_url = excluded.storage_url,
                    total_pages = excluded.total_pages,
                    total_elements = excluded.total_elements,
                    metadata = excluded.metadata,
                    created_at = now()
                """,
                (document_id, kind, storage_url, total_pages, total_elements, metadata_param),
            )

    def get_extractions(self, document_id: uuid.UUID) -> List[Dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                select id, kind, storage_url, total_pages, total_elements, metadata, created_at
                from extractions
                where document_id = %s
                order by created_at desc
                """,
                (document_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r[0]),
                    "kind": r[1],
                    "storage_url": r[2],
                    "total_pages": r[3],
                    "total_elements": r[4],
                    "metadata": r[5],
                    "created_at": r[6].isoformat(),
                }
                for r in rows
            ]
