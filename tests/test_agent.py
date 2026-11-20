import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openhaven.agent import OpenHavenAgent  # noqa: E402
from openhaven.llm import ScriptedLLM  # noqa: E402
from openhaven.models import AgentRequest, CandidateProfile, Evidence, Style, Workflow  # noqa: E402
from openhaven.storage import JsonStateStore  # noqa: E402


class OpenHavenAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = CandidateProfile(
            candidate_name=None,
            evidence=[
                Evidence(
                    id="education.major",
                    category="education",
                    text="本科专业为计算机科学与技术。",
                )
            ],
        )

    def test_retries_an_overlong_form_answer(self) -> None:
        llm = ScriptedLLM(
            [
                {
                    "sections": [
                        {
                            "title": "任意标题",
                            "blocks": [
                                {
                                    "text": "我的本科专业为计算机科学与技术。",
                                    "kind": "candidate_fact",
                                    "evidence_ids": ["education.major"],
                                }
                            ],
                        }
                    ]
                },
                {
                    "sections": [
                        {
                            "title": "任意标题",
                            "compressed": True,
                            "blocks": [
                                {
                                    "text": "本科计算机",
                                    "kind": "candidate_fact",
                                    "evidence_ids": ["education.major"],
                                }
                            ],
                        }
                    ]
                },
            ]
        )
        request = AgentRequest(
            workflow=Workflow.FORM_ANSWERS,
            style=Style.TECHNICAL,
            payload={"questions": [{"label": "所学专业", "max_len": 5}]},
        )
        response = OpenHavenAgent(self.profile, llm=llm).run(request)
        self.assertEqual(response.sections[0].title, "所学专业")
        self.assertEqual(response.sections[0].text, "本科计算机")
        self.assertTrue(response.sections[0].compressed)
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("length_exceeded", llm.prompts[1].user)

    def test_dry_run_contains_only_supplied_evidence(self) -> None:
        request = AgentRequest(
            workflow=Workflow.INTERVIEW_KIT,
            style=Style.INSTITUTIONAL,
            payload={"duration": "1分钟"},
        )
        prompt = OpenHavenAgent(self.profile).prepare(request)
        self.assertIn("本科专业为计算机科学与技术", prompt.user)
        self.assertIn("稳重克制", prompt.system)
        self.assertIn("不得补全、猜测或虚构", prompt.system)

    def test_requires_jd_for_cover_letter(self) -> None:
        request = AgentRequest(
            workflow=Workflow.COVER_LETTER,
            style=Style.TECHNICAL,
            payload={},
        )
        with self.assertRaises(ValueError):
            OpenHavenAgent(self.profile).prepare(request)

    def test_records_application_without_an_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = JsonStateStore(Path(directory) / "state.json")
            request = AgentRequest(
                workflow=Workflow.APPLICATION_LEDGER,
                style=Style.TECHNICAL,
                application_id="app-1",
                payload={
                    "action": "record",
                    "record": {
                        "company": "示例公司",
                        "role": "后端开发",
                        "jd_link": None,
                        "applied_at": "2026-10-28",
                        "match_score": 80,
                        "role_type": "技术岗",
                        "status": "已投递",
                    },
                },
            )
            response = OpenHavenAgent(self.profile, state_store=state).run(request)
            self.assertIn("示例公司", response.sections[0].text)
            self.assertEqual(len(state.list_applications()), 1)


if __name__ == "__main__":
    unittest.main()

