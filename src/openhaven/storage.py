from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


VALID_APPLICATION_STATUSES = {
    "待投递",
    "已投递",
    "初面",
    "二面",
    "挂掉",
    "offer",
}


@dataclass(frozen=True, slots=True)
class ApplicationRecord:
    application_id: str
    company: str
    role: str
    jd_link: str | None
    applied_at: str
    match_score: int | None
    role_type: str
    status: str
    notes: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ApplicationRecord":
        score = value.get("match_score")
        if score is not None and (not isinstance(score, int) or not 0 <= score <= 100):
            raise ValueError("match_score must be an integer between 0 and 100")
        status = str(value.get("status", "待投递"))
        if status not in VALID_APPLICATION_STATUSES:
            raise ValueError(f"unsupported application status: {status}")
        applied_at = str(value["applied_at"])
        date.fromisoformat(applied_at)
        return cls(
            application_id=str(value["application_id"]),
            company=str(value["company"]),
            role=str(value["role"]),
            jd_link=value.get("jd_link"),
            applied_at=applied_at,
            match_score=score,
            role_type=str(value["role_type"]),
            status=status,
            notes=value.get("notes"),
        )


class JsonStateStore:
    """Small local store for explicit approvals and application records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get_confirmed_context(self, application_id: str | None) -> dict[str, Any]:
        if not application_id:
            return {}
        data = self._read()
        return dict(data.get("confirmed_context", {}).get(application_id, {}))

    def confirm_context(self, application_id: str, context: dict[str, Any]) -> None:
        """Persist context only after an explicit human approval action."""
        data = self._read()
        data.setdefault("confirmed_context", {})[application_id] = context
        self._write(data)

    def upsert_application(self, record: ApplicationRecord) -> None:
        data = self._read()
        data.setdefault("applications", {})[record.application_id] = asdict(record)
        self._write(data)

    def list_applications(self) -> list[ApplicationRecord]:
        values = self._read().get("applications", {}).values()
        return [ApplicationRecord.from_dict(item) for item in values]

    def summarize_applications(
        self,
        today: date | None = None,
        stale_after_days: int = 14,
    ) -> dict[str, Any]:
        current = today or date.today()
        records = self.list_applications()
        status_counts = Counter(item.status for item in records)
        stale = []
        for item in records:
            if item.status not in {"已投递", "初面", "二面"}:
                continue
            age = (current - date.fromisoformat(item.applied_at)).days
            if age >= stale_after_days:
                stale.append(
                    {
                        "application_id": item.application_id,
                        "company": item.company,
                        "role": item.role,
                        "status": item.status,
                        "days_without_update": age,
                    }
                )
        return {
            "total": len(records),
            "status_counts": dict(status_counts),
            "stale_applications": stale,
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "confirmed_context": {}, "applications": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("state file must contain a JSON object")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def parse_iso_datetime(value: str) -> datetime:
    """Reserved for future event history while keeping ISO parsing centralized."""
    return datetime.fromisoformat(value)

