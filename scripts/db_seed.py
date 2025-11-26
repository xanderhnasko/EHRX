#!/usr/bin/env python
"""
Initialize DB schema and optionally seed a sample extraction record.

Examples:
    python scripts/db_seed.py
    python scripts/db_seed.py \
        --sample-full output/test_20_pages/SENSITIVE_ehr1_copy_1763164390_full.json \
        --bucket-url gs://ehrx-artifacts
"""

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

# Ensure project root is importable when run from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ehrx.db.config import DBConfig
from ehrx.db.client import get_conn, init_schema


def seed_sample(conn, full_path: Path, bucket_url: str | None) -> None:
    """Insert a document + full extraction row."""
    storage_url = f"{bucket_url}/{full_path.name}" if bucket_url else str(full_path)

    with open(full_path, "r") as f:
        doc_json = json.load(f)

    total_pages = doc_json.get("total_pages")
    total_elements = doc_json.get("processing_stats", {}).get("total_elements")

    with conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into documents (original_filename, storage_url, sha256)
            values (%s, %s, %s)
            returning id
            """,
            (full_path.name, storage_url, None),
        )
        doc_id = cur.fetchone()[0]

        cur.execute(
            """
            insert into extractions
                (document_id, kind, storage_url, total_pages, total_elements, metadata)
            values
                (%s, 'full', %s, %s, %s, %s)
            """,
            (
                doc_id,
                storage_url,
                total_pages,
                total_elements,
                json.dumps({"local_seed": True}),
            ),
        )

    print(f"Seeded document {doc_id} -> {storage_url}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Initialize DB schema and optionally seed sample data.")
    parser.add_argument("--sample-full", help="Path to *_full.json to register (optional)")
    parser.add_argument("--bucket-url", help="gs:// bucket root for storage_url (optional)")
    args = parser.parse_args()

    cfg = DBConfig.from_env()

    with get_conn(cfg) as conn:
        init_schema(conn)
        print("Schema ensured (documents, extractions).")

        if args.sample_full:
            full_path = Path(args.sample_full)
            if not full_path.exists():
                raise FileNotFoundError(f"Sample file not found: {full_path}")
            seed_sample(conn, full_path, args.bucket_url)


if __name__ == "__main__":
    main()
