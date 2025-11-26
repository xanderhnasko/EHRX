"""
Database configuration helpers.

Reads connection settings from environment. Password should come from
Secret Manager in deployed environments; .env is acceptable for local dev.
"""

import os
from dataclasses import dataclass


@dataclass
class DBConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    sslmode: str = "require"

    @classmethod
    def from_env(cls) -> "DBConfig":
        return cls(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "ehrx"),
            user=os.getenv("DB_USER", "appuser"),
            password=os.getenv("DB_PASSWORD", ""),
            sslmode=os.getenv("DB_SSLMODE", "require"),
        )
