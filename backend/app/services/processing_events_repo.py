from __future__ import annotations

from typing import Any, Dict, Optional

from app.services._supabase_rest import insert_row


class ProcessingEventsRepo:
    async def log(
        self,
        *,
        user_id: str,
        job_id: str,
        file_id: str,
        event_type: str,
        severity: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        await insert_row(
            table="processing_events",
            row={
                "user_id": user_id,
                "job_id": job_id,
                "file_id": file_id,
                "event_type": event_type,
                "severity": severity,
                "details": details or {},
            },
        )
