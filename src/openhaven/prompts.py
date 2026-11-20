from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .models import AgentRequest, CandidateProfile, Evidence, Style, Workflow


@dataclass(frozen=True, slots=True)
class PromptPackage:
    system: str
    user: str
    schema_hint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "user": self.user,
            "schema_hint": self.schema_hint,
        }


BASE_RULES = """你是 OpenHaven 求职材料加工智能体，不是自动投递机器人。

不可突破的规则：
1. 候选人事实只能来自 <candidate_evidence>；JD 事实只能来自 <jd_evidence>。不得补全、猜测或虚构。
2. 每一个候选人事实块必须标记 kind=candidate_fact，并引用支持它的 candidate evidence_ids。
3. 每一个 JD 事实块必须标记 kind=jd_fact，并引用支持它的 jd evidence_ids。
4. 分析、建议和匹配分分别使用 kind=analysis 或 advice；不得在这两类块里暗藏新的候选人经历。
5. 素材不足时直接输出“【素材无相关经历，建议填无】”，不得强行包装。
6. 不声称已经访问网页、填写表单、上传附件、联系 HR 或完成投递。
7. 不生成夸大造假、不符合政审或招聘规范的话术。军队、公职、国企事业单位用词必须保守。
8. 同一 application_id 下，以 prior_context 中已经确认的口径为准，不得改动时间、职责或成果。
9. 字数上限按非空白可见字符计算。回答正文必须在 max_len 内；字数标签不写入正文。
10. 只返回符合 schema 的 JSON，不输出代码围栏或额外解释。"""

STYLE_RULES = {
    Style.TECHNICAL: """采用技术赛道风格：简洁、具体、少空话。优先呈现技术链路、工具栈、工程问题、行动和已有量化结果；可用 STAR 组织，但不能补造工具或指标。""",
    Style.INSTITUTIONAL: """采用体制赛道风格：稳重克制、正向合规、措辞严谨。突出已有材料支持的责任心、学习能力、科研态度和协作意识；避免互联网黑话、炫技和夸张承诺。""",
}

WORKFLOW_RULES = {
    Workflow.JD_ANALYSIS: """抽取硬性门槛、加分项、业务方向和软素质；对照素材输出匹配点、短板、可侧重的已有亮点、0—100 匹配分及理由、关键词和推荐文风。短板必须诚实。""",
    Workflow.FORM_ANSWERS: """逐题识别考察意图并作答。sections 数量、顺序和标题必须与 questions 一致。每题遵守 max_len。敏感或政审问题使用稳妥表述。""",
    Workflow.RESUME_REWRITE: """按 JD 重构用户指定的简历段落。技术岗强调技术链路和工程落地；体制岗强调目标、职责、完成工作和价值。只能重组已有素材。""",
    Workflow.INTERVIEW_KIT: """生成用户指定时长的自我介绍、高频通用题、技术深挖题、跨专业问题及参考回答，并给出三个合适的反问问题。参考回答仍须受素材约束。""",
    Workflow.APPLICATION_LEDGER: """仅对用户提供的台账数据做汇总，统计阶段分布和长时间无反馈项，并生成温和跟进话术。不得声称已联系 HR。""",
    Workflow.COVER_LETTER: """生成简短、定向的求职信或自荐信，绑定 JD 和候选人已有材料，避免通用模板堆砌，并遵守 max_len。""",
}

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["sections"],
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "blocks"],
                "properties": {
                    "title": {"type": "string"},
                    "compressed": {"type": "boolean"},
                    "blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["text", "kind", "evidence_ids"],
                            "properties": {
                                "text": {"type": "string"},
                                "kind": {
                                    "enum": [
                                        "candidate_fact",
                                        "jd_fact",
                                        "analysis",
                                        "advice",
                                        "label",
                                    ]
                                },
                                "evidence_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def build_prompt_package(
    request: AgentRequest,
    profile: CandidateProfile,
    jd_evidence: list[Evidence],
    prior_context: dict[str, Any] | None = None,
    repair_context: dict[str, Any] | None = None,
) -> PromptPackage:
    system = "\n\n".join(
        [BASE_RULES, STYLE_RULES[request.style], WORKFLOW_RULES[request.workflow]]
    )
    body = {
        "workflow": request.workflow.value,
        "style": request.style.value,
        "application_id": request.application_id,
        "plain_text": request.plain_text,
        "candidate_evidence": [
            {
                "id": item.id,
                "category": item.category,
                "text": item.text,
            }
            for item in profile.evidence
        ],
        "candidate_preferences": {
            "target_roles": profile.target_roles,
            "preferred_regions": profile.preferred_regions,
            "avoided_roles": profile.avoided_roles,
            "travel_preference": profile.travel_preference,
        },
        "jd_evidence": [
            {"id": item.id, "category": item.category, "text": item.text}
            for item in jd_evidence
        ],
        "prior_context": prior_context or {},
        "task_payload": request.payload,
    }
    if repair_context:
        body["repair"] = repair_context
    user = json.dumps(body, ensure_ascii=False, indent=2)
    return PromptPackage(system=system, user=user, schema_hint=RESPONSE_SCHEMA)


def jd_to_evidence(jd_text: str) -> list[Evidence]:
    chunks = [line.strip(" -•\t") for line in jd_text.splitlines() if line.strip()]
    if not chunks and jd_text.strip():
        chunks = [jd_text.strip()]
    return [
        Evidence(
            id=f"jd.{index:03d}",
            category="job_requirement",
            text=chunk,
            origin="jd",
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

