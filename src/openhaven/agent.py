from __future__ import annotations

from dataclasses import replace
from typing import Any

from .llm import LLMClient
from .models import (
    AgentRequest,
    AgentResponse,
    BlockKind,
    CandidateProfile,
    FormQuestion,
    GroundedBlock,
    ResponseSection,
    Workflow,
)
from .policy import GuardrailPolicy, PolicyIssue, PolicyViolation
from .prompts import PromptPackage, build_prompt_package, jd_to_evidence
from .storage import ApplicationRecord, JsonStateStore


class OpenHavenAgent:
    def __init__(
        self,
        profile: CandidateProfile,
        llm: LLMClient | None = None,
        policy: GuardrailPolicy | None = None,
        state_store: JsonStateStore | None = None,
        max_attempts: int = 2,
    ):
        self.profile = profile
        self.llm = llm
        self.policy = policy or GuardrailPolicy()
        self.state_store = state_store
        self.max_attempts = max(1, max_attempts)
        self.policy.validate_profile(profile)

    def prepare(self, request: AgentRequest) -> PromptPackage:
        self._validate_request(request)
        jd_evidence = jd_to_evidence(str(request.payload.get("jd_text", "")))
        prior = (
            self.state_store.get_confirmed_context(request.application_id)
            if self.state_store
            else {}
        )
        return build_prompt_package(request, self.profile, jd_evidence, prior)

    def run(self, request: AgentRequest) -> AgentResponse:
        self._validate_request(request)
        if request.workflow == Workflow.APPLICATION_LEDGER:
            return self._run_ledger(request)
        if self.llm is None:
            raise RuntimeError("未配置模型客户端；可先调用 prepare() 查看请求包")

        jd_evidence = jd_to_evidence(str(request.payload.get("jd_text", "")))
        evidence = self.profile.evidence_map() | {item.id: item for item in jd_evidence}
        prior = (
            self.state_store.get_confirmed_context(request.application_id)
            if self.state_store
            else {}
        )
        repair_context: dict[str, Any] | None = None
        last_issues: tuple[PolicyIssue, ...] = ()

        for _ in range(self.max_attempts):
            prompt = build_prompt_package(
                request,
                self.profile,
                jd_evidence,
                prior_context=prior,
                repair_context=repair_context,
            )
            raw = self.llm.complete_json(prompt)
            sections, structure_issues = self._parse_and_bind_sections(request, raw)
            policy_issues = self.policy.validate_sections(sections, evidence)
            last_issues = tuple(structure_issues) + tuple(policy_issues)
            if not last_issues:
                return AgentResponse(
                    workflow=request.workflow,
                    style=request.style,
                    sections=tuple(sections),
                    warnings=tuple(str(item) for item in raw.get("warnings", [])),
                )
            repair_context = {
                "instruction": "修复以下问题；不得通过删除素材引用或改成分析块来规避事实校验。",
                "issues": [
                    {"code": item.code, "message": item.message, "section": item.section}
                    for item in last_issues
                ],
                "previous_response": raw,
            }

        raise PolicyViolation(last_issues)

    def _run_ledger(self, request: AgentRequest) -> AgentResponse:
        if self.state_store is None:
            raise ValueError("application_ledger 需要配置本地 state_store")
        action = str(request.payload.get("action", "summary"))
        if action == "record":
            value = request.payload.get("record")
            if not isinstance(value, dict):
                raise ValueError("记录投递需要 payload.record")
            if request.application_id and "application_id" not in value:
                value = {**value, "application_id": request.application_id}
            record = ApplicationRecord.from_dict(value)
            self.state_store.upsert_application(record)
            text = f"已记录：{record.company}｜{record.role}｜{record.status}｜{record.applied_at}"
            sections = (
                ResponseSection(
                    title="投递记录",
                    blocks=(GroundedBlock(text=text, kind=BlockKind.ANALYSIS),),
                ),
            )
        elif action == "summary":
            stale_after_days = request.payload.get("stale_after_days", 14)
            if not isinstance(stale_after_days, int) or stale_after_days <= 0:
                raise ValueError("stale_after_days 必须是正整数")
            summary = self.state_store.summarize_applications(
                stale_after_days=stale_after_days
            )
            counts = "、".join(
                f"{status}{count}个"
                for status, count in sorted(summary["status_counts"].items())
            ) or "暂无"
            stale_lines = [
                f"{item['company']}｜{item['role']}｜{item['status']}｜{item['days_without_update']}天未更新"
                for item in summary["stale_applications"]
            ]
            stale_text = "\n".join(stale_lines) or "无"
            sections = (
                ResponseSection(
                    title="投递概览",
                    blocks=(
                        GroundedBlock(
                            text=f"累计{summary['total']}个；阶段分布：{counts}。",
                            kind=BlockKind.ANALYSIS,
                        ),
                    ),
                ),
                ResponseSection(
                    title="长时间无反馈",
                    blocks=(GroundedBlock(text=stale_text, kind=BlockKind.ANALYSIS),),
                ),
                ResponseSection(
                    title="跟进话术模板",
                    blocks=(
                        GroundedBlock(
                            text="您好，我此前申请了贵单位的相关岗位，想礼貌了解目前的招聘进展。如需补充材料，我会及时配合。感谢您的时间。",
                            kind=BlockKind.ADVICE,
                        ),
                    ),
                ),
            )
        else:
            raise ValueError("application_ledger action 仅支持 record 或 summary")

        return AgentResponse(
            workflow=request.workflow,
            style=request.style,
            sections=sections,
        )

    def _validate_request(self, request: AgentRequest) -> None:
        action = str(request.payload.get("action", request.workflow.value))
        self.policy.assert_action_allowed(action)
        if request.workflow in {
            Workflow.JD_ANALYSIS,
            Workflow.RESUME_REWRITE,
            Workflow.COVER_LETTER,
        } and not str(request.payload.get("jd_text", "")).strip():
            raise ValueError(f"{request.workflow.value} 需要 payload.jd_text")
        if request.workflow == Workflow.FORM_ANSWERS:
            questions = request.payload.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError("form_answers 需要非空 payload.questions")
            for item in questions:
                FormQuestion.from_dict(item)

    def _parse_and_bind_sections(
        self,
        request: AgentRequest,
        raw: dict[str, Any],
    ) -> tuple[list[ResponseSection], tuple[PolicyIssue, ...]]:
        raw_sections = raw.get("sections", [])
        if not isinstance(raw_sections, list):
            return [], (PolicyIssue("invalid_schema", "sections 必须是数组"),)
        try:
            sections = [ResponseSection.from_dict(item) for item in raw_sections]
        except (TypeError, ValueError) as exc:
            return [], (PolicyIssue("invalid_schema", f"响应结构无法解析：{exc}"),)

        issues: list[PolicyIssue] = []
        if not sections:
            issues.append(PolicyIssue("empty_response", "模型未返回任何回答段落"))
            return sections, tuple(issues)

        if request.workflow == Workflow.FORM_ANSWERS:
            questions = [
                FormQuestion.from_dict(item) for item in request.payload["questions"]
            ]
            if len(sections) != len(questions):
                issues.append(
                    PolicyIssue(
                        "question_count_mismatch",
                        f"应回答 {len(questions)} 题，实际返回 {len(sections)} 题",
                    )
                )
            for index, question in enumerate(questions[: len(sections)]):
                sections[index] = replace(
                    sections[index], title=question.label, max_len=question.max_len
                )
        else:
            requested_max = request.payload.get("max_len")
            if requested_max is not None:
                if not isinstance(requested_max, int) or requested_max <= 0:
                    issues.append(
                        PolicyIssue("invalid_max_len", "max_len 必须是正整数")
                    )
                elif len(sections) == 1:
                    sections[0] = replace(sections[0], max_len=requested_max)
                else:
                    issues.append(
                        PolicyIssue(
                            "ambiguous_max_len",
                            "多段输出请在请求中为每段分别指定字数上限",
                        )
                    )
        return sections, tuple(issues)

