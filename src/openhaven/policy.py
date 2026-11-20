from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from .models import BlockKind, CandidateProfile, Evidence, ResponseSection, visible_char_count


FORBIDDEN_ACTION_TERMS = (
    "browser",
    "click",
    "crawl",
    "crawler",
    "scrape",
    "submit",
    "upload",
    "浏览器",
    "点击",
    "爬虫",
    "抓取",
    "提交表单",
    "自动投递",
    "上传附件",
)

SENSITIVE_CREDENTIAL_TERMS = (
    "论文",
    "发表",
    "专利",
    "软著",
    "奖学金",
    "一等奖",
    "二等奖",
    "三等奖",
    "证书",
    "认证",
    "四六级",
    "CET-4",
    "CET-6",
    "党员",
    "学生会",
    "班干部",
    "实习",
    "offer",
)

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?")


@dataclass(frozen=True, slots=True)
class PolicyIssue:
    code: str
    message: str
    section: str | None = None


class PolicyViolation(ValueError):
    def __init__(self, issues: Iterable[PolicyIssue]):
        self.issues = tuple(issues)
        message = "; ".join(issue.message for issue in self.issues)
        super().__init__(message)


class GuardrailPolicy:
    """Deterministic checks performed after every model response."""

    def assert_action_allowed(self, action: str) -> None:
        lowered = action.casefold()
        if any(term.casefold() in lowered for term in FORBIDDEN_ACTION_TERMS):
            raise PolicyViolation(
                [PolicyIssue("forbidden_action", f"OpenHaven 不执行该动作：{action}")]
            )

    def validate_profile(self, profile: CandidateProfile) -> None:
        ids = [item.id for item in profile.evidence]
        issues: list[PolicyIssue] = []
        if len(ids) != len(set(ids)):
            issues.append(PolicyIssue("duplicate_evidence_id", "素材 ID 必须唯一"))
        for item in profile.evidence:
            if not item.id.strip() or not item.text.strip():
                issues.append(
                    PolicyIssue("empty_evidence", "素材 ID 和内容均不能为空")
                )
        if issues:
            raise PolicyViolation(issues)

    def validate_sections(
        self,
        sections: Iterable[ResponseSection],
        evidence: dict[str, Evidence],
    ) -> tuple[PolicyIssue, ...]:
        issues: list[PolicyIssue] = []
        candidate_corpus = "\n".join(
            item.text for item in evidence.values() if item.origin == "candidate"
        )

        for section in sections:
            if section.max_len is not None and visible_char_count(section.text) > section.max_len:
                issues.append(
                    PolicyIssue(
                        "length_exceeded",
                        f"“{section.title}”为 {visible_char_count(section.text)} 字，超过 {section.max_len} 字上限",
                        section.title,
                    )
                )

            for block in section.blocks:
                missing = [item for item in block.evidence_ids if item not in evidence]
                if missing:
                    issues.append(
                        PolicyIssue(
                            "unknown_evidence",
                            f"“{section.title}”引用不存在的素材：{', '.join(missing)}",
                            section.title,
                        )
                    )
                    continue

                if block.kind in (BlockKind.CANDIDATE_FACT, BlockKind.JD_FACT):
                    if not block.evidence_ids:
                        issues.append(
                            PolicyIssue(
                                "ungrounded_fact",
                                f"“{section.title}”包含未引用素材的事实陈述",
                                section.title,
                            )
                        )
                        continue

                    expected_origin = (
                        "candidate" if block.kind == BlockKind.CANDIDATE_FACT else "jd"
                    )
                    linked = [evidence[item] for item in block.evidence_ids]
                    if not any(item.origin == expected_origin for item in linked):
                        issues.append(
                            PolicyIssue(
                                "wrong_evidence_origin",
                                f"“{section.title}”的事实引用来源类型不正确",
                                section.title,
                            )
                        )

                    source_text = " ".join(item.text for item in linked)
                    issues.extend(
                        self._validate_grounded_block(
                            block.text,
                            source_text,
                            candidate_corpus,
                            section.title,
                            block.kind,
                        )
                    )

        return tuple(issues)

    def require_valid_sections(
        self,
        sections: Iterable[ResponseSection],
        evidence: dict[str, Evidence],
    ) -> None:
        issues = self.validate_sections(sections, evidence)
        if issues:
            raise PolicyViolation(issues)

    def _validate_grounded_block(
        self,
        text: str,
        source_text: str,
        candidate_corpus: str,
        section_title: str,
        kind: BlockKind,
    ) -> list[PolicyIssue]:
        issues: list[PolicyIssue] = []
        if not text.strip():
            return [PolicyIssue("empty_block", "模型返回了空文本块", section_title)]

        if kind == BlockKind.CANDIDATE_FACT:
            source_numbers = set(NUMBER_PATTERN.findall(source_text))
            output_numbers = set(NUMBER_PATTERN.findall(text))
            invented_numbers = sorted(output_numbers - source_numbers)
            if invented_numbers:
                issues.append(
                    PolicyIssue(
                        "invented_number",
                        f"“{section_title}”出现素材未支持的数字：{', '.join(invented_numbers)}",
                        section_title,
                    )
                )

            for term in SENSITIVE_CREDENTIAL_TERMS:
                if term.casefold() in text.casefold() and term.casefold() not in candidate_corpus.casefold():
                    issues.append(
                        PolicyIssue(
                            "invented_credential",
                            f"“{section_title}”出现素材未支持的敏感资历词：{term}",
                            section_title,
                        )
                    )

        compact_output = re.sub(r"\s+", "", text)
        compact_source = re.sub(r"\s+", "", source_text)
        if len(compact_output) >= 12 and len(compact_source) >= 6:
            similarity = SequenceMatcher(None, compact_output, compact_source).ratio()
            shared_chars = set(compact_output) & set(compact_source)
            if similarity < 0.08 and len(shared_chars) < 4:
                issues.append(
                    PolicyIssue(
                        "weak_grounding",
                        f"“{section_title}”与其引用素材缺少可验证的文本关联",
                        section_title,
                    )
                )
        return issues

