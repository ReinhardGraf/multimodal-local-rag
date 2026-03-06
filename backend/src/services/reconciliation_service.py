import os
from typing import List, Dict, Any
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.config import settings
import asyncpg


QDRANT_URL = settings.qdrant_url
QDRANT_COLLECTION = settings.qdrant_collection_name
POSTGRES_DSN = settings.postgres_dsn

class ReconciliationService:


    async def ensure_qdrant_indexes(self, client: QdrantClient):
        fields = [
            ("fileHash", rest.PayloadSchemaType.KEYWORD),
            ("source", rest.PayloadSchemaType.KEYWORD),
            ("contentType", rest.PayloadSchemaType.KEYWORD),
            ("fileSize", rest.PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema in fields:
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                # Index already exists → ignore
                pass


    async def reconcile_file_hashes(self) -> Dict[str, Any]:
        qdrant = QdrantClient(url=QDRANT_URL)
        pg = await asyncpg.connect(dsn=POSTGRES_DSN)

        await self.ensure_qdrant_indexes(qdrant)

        # 1️⃣ Get distinct fileHash values (server-side aggregation)
        facet_response = qdrant.facet(
            collection_name=QDRANT_COLLECTION,
            key="fileHash",
            limit=100_000,
        )

        hashes = [entry.value for entry in facet_response.hits]

        processed = 0

        for file_hash in hashes:

            # 2️⃣ Count chunks server-side
            count_response = qdrant.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="fileHash",
                            match=rest.MatchValue(value=file_hash),
                        )
                    ]
                ),
            )

            chunk_count = count_response.count

            if chunk_count == 0:
                continue

            # 3️⃣ Fetch representative payload
            points, _ = qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="fileHash",
                            match=rest.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                continue

            payload = points[0].payload

            file_path = payload.get("source")
            file_size = payload.get("fileSize")
            content_type = payload.get("contentType")

            if not file_path:
                continue

            # 4️⃣ Idempotent UPSERT
            await pg.execute(
                """
                INSERT INTO file_hashes 
                    (file_path, file_hash, chunk_count, file_size, content_type, processed_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (file_path) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    chunk_count = EXCLUDED.chunk_count,
                    file_size = EXCLUDED.file_size,
                    content_type = EXCLUDED.content_type,
                    processed_at = NOW(),
                    last_error = NULL
                """,
                file_path,
                file_hash,
                chunk_count,
                file_size,
                content_type,
            )

            processed += 1

        await pg.close()

        return {
            "status": "success",
            "distinct_hashes": len(hashes),
            "processed": processed,
            "timestamp": datetime.utcnow().isoformat(),
        }