"""Store and search embeddings in PostgreSQL with pgvector."""
import json
import math
from typing import Any

import psycopg

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be a non-empty string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")

        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the persistent chunk table and metadata indexes."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_strategy_document_idx
                    ON chunks (strategy, document_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
                    ON chunks USING GIN (metadata)
                    """
                )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace the stored chunks for documents represented in this batch.

        Other documents and strategies are left untouched. Empty batches are no-ops.
        """
        self._validate_strategy(strategy)
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        rows: list[tuple[str, str, str, str, int, str, str, str]] = []
        document_ids: set[str] = set()
        chunk_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            self._validate_required_text(chunk.chunk_id, "chunk_id")
            self._validate_required_text(chunk.document_id, "document_id")
            self._validate_required_text(chunk.source, "source")
            if not isinstance(chunk.text, str):
                raise ValueError(f"chunk {chunk.chunk_id!r} text must be a string")
            if isinstance(chunk.page, bool) or not isinstance(chunk.page, int):
                raise ValueError(f"chunk {chunk.chunk_id!r} page must be an integer")
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"duplicate chunk_id in input: {chunk.chunk_id!r}")
            chunk_ids.add(chunk.chunk_id)
            document_ids.add(chunk.document_id)

            metadata_json = self._serialize_json(chunk.metadata, "chunk metadata")
            vector_literal = self._vector_literal(vector, f"chunk {chunk.chunk_id!r} vector")
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    metadata_json,
                    vector_literal,
                )
            )

        insert_sql = """
            INSERT INTO chunks
                (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM chunks
                    WHERE strategy = %s AND document_id = ANY(%s)
                    """,
                    (strategy, sorted(document_ids)),
                )
                cursor.executemany(insert_sql, rows)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search by cosine similarity, optionally restricting JSONB metadata.

        Filters use JSONB containment, so each provided key/value must be present
        in a stored chunk's metadata.
        """
        self._validate_strategy(strategy)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        vector_literal = self._vector_literal(query_vector, "query vector")
        where_sql = "strategy = %s"
        parameters: list[Any] = [vector_literal, strategy]
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError("filters must be a dictionary")
            if filters:
                where_sql += " AND metadata @> %s::jsonb"
                parameters.append(self._serialize_json(filters, "filters"))
        parameters.extend((vector_literal, top_k))

        query = f"""
            SELECT chunk_id, document_id, source, page, text,
                   1 - (embedding <=> %s::vector) AS score, metadata
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                rows = cursor.fetchall()

        return [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source,
                "page": page,
                "text": text,
                "score": float(score),
                "metadata": metadata,
            }
            for chunk_id, document_id, source, page, text, score, metadata in rows
        ]

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        PgVectorStore._validate_required_text(strategy, "strategy")

    @staticmethod
    def _validate_required_text(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    def _vector_literal(self, vector: list[float], name: str) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"{name} has {len(vector)} dimensions; expected {self.dimensions}"
            )
        values: list[str] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} values must be finite numbers")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name} values must be finite numbers")
            values.append(repr(numeric_value))
        return f"[{','.join(values)}]"

    @staticmethod
    def _serialize_json(value: Any, name: str) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be JSON serializable") from error
