import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openhaven.models import (  # noqa: E402
    BlockKind,
    CandidateProfile,
    Evidence,
    GroundedBlock,
    ResponseSection,
)
from openhaven.policy import GuardrailPolicy, PolicyViolation  # noqa: E402


class GuardrailPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GuardrailPolicy()
        self.profile = CandidateProfile(
            candidate_name=None,
            evidence=[
                Evidence(
                    id="project.latency",
                    category="project",
                    text="将接口延迟降低20%。",
                ),
                Evidence(
                    id="education.major",
                    category="education",
                    text="本科专业为计算机科学与技术。",
                ),
            ],
        )
        self.evidence = self.profile.evidence_map()

    def test_accepts_grounded_candidate_fact(self) -> None:
        section = ResponseSection(
            title="项目经历",
            blocks=(
                GroundedBlock(
                    text="通过优化使接口延迟降低20%。",
                    kind=BlockKind.CANDIDATE_FACT,
                    evidence_ids=("project.latency",),
                ),
            ),
            max_len=50,
        )
        self.assertEqual(self.policy.validate_sections([section], self.evidence), ())

    def test_rejects_invented_number(self) -> None:
        section = ResponseSection(
            title="项目经历",
            blocks=(
                GroundedBlock(
                    text="通过优化使接口延迟降低50%。",
                    kind=BlockKind.CANDIDATE_FACT,
                    evidence_ids=("project.latency",),
                ),
            ),
        )
        issues = self.policy.validate_sections([section], self.evidence)
        self.assertIn("invented_number", {item.code for item in issues})

    def test_rejects_invented_credential(self) -> None:
        section = ResponseSection(
            title="个人优势",
            blocks=(
                GroundedBlock(
                    text="本科阶段发表论文。",
                    kind=BlockKind.CANDIDATE_FACT,
                    evidence_ids=("education.major",),
                ),
            ),
        )
        issues = self.policy.validate_sections([section], self.evidence)
        self.assertIn("invented_credential", {item.code for item in issues})

    def test_enforces_visible_character_limit(self) -> None:
        section = ResponseSection(
            title="回答",
            blocks=(
                GroundedBlock(
                    text="本科专业为计算机科学与技术。",
                    kind=BlockKind.CANDIDATE_FACT,
                    evidence_ids=("education.major",),
                ),
            ),
            max_len=5,
        )
        issues = self.policy.validate_sections([section], self.evidence)
        self.assertIn("length_exceeded", {item.code for item in issues})

    def test_rejects_forbidden_action(self) -> None:
        with self.assertRaises(PolicyViolation):
            self.policy.assert_action_allowed("自动投递这个岗位")


if __name__ == "__main__":
    unittest.main()

