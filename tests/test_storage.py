import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openhaven.storage import ApplicationRecord, JsonStateStore  # noqa: E402


class JsonStateStoreTests(unittest.TestCase):
    def test_context_is_empty_until_explicitly_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonStateStore(Path(directory) / "state.json")
            self.assertEqual(store.get_confirmed_context("app-1"), {})
            store.confirm_context("app-1", {"research_time": "2025-2026"})
            self.assertEqual(
                store.get_confirmed_context("app-1"),
                {"research_time": "2025-2026"},
            )

    def test_summarizes_stale_applications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonStateStore(Path(directory) / "state.json")
            store.upsert_application(
                ApplicationRecord(
                    application_id="app-1",
                    company="示例公司",
                    role="后端开发",
                    jd_link=None,
                    applied_at="2026-10-01",
                    match_score=80,
                    role_type="技术岗",
                    status="已投递",
                )
            )
            summary = store.summarize_applications(today=date(2026, 10, 28))
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["status_counts"], {"已投递": 1})
            self.assertEqual(summary["stale_applications"][0]["days_without_update"], 27)


if __name__ == "__main__":
    unittest.main()

