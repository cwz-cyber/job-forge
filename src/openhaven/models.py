from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Style(StrEnum):
    TECHNICAL = "technical"
    INSTITUTIONAL = "institutional"


class Workflow(StrEnum):
    JD_ANALYSIS = "jd_analysis"
    FORM_ANSWERS = "form_answers"
    RESUME_REWRITE = "resume_rewrite"
    INTERVIEW_KIT = "interview_kit"
    APPLICATION_LEDGER = "application_ledger"
    COVER_LETTER = "cover_letter"


class BlockKind(StrEnum):
    CANDIDATE_FACT = "candidate_fact"
    JD_FACT = "jd_fact"
    ANALYSIS = "analysis"
    ADVICE = "advice"
    LABEL = "label"


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    category: str
    text: str
    origin: str = "candidate"
    source_note: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Evidence":
        return cls(
            id=str(value["id"]),
            category=str(value["category"]),
            text=str(value["text"]),
            origin=str(value.get("origin", "candidate")),
            source_note=value.get("source_note"),
        )


@dataclass(slots=True)
class CandidateProfile:
    candidate_name: str | None
    evidence: list[Evidence]
    target_roles: list[str] = field(default_factory=list)
    preferred_regions: list[str] = field(default_factory=list)
    avoided_roles: list[str] = field(default_factory=list)
    travel_preference: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateProfile":
        return cls(
            candidate_name=value.get("candidate_name"),
            evidence=[Evidence.from_dict(item) for item in value.get("evidence", [])],
            target_roles=[str(item) for item in value.get("target_roles", [])],
            preferred_regions=[str(item) for item in value.get("preferred_regions", [])],
            avoided_roles=[str(item) for item in value.get("avoided_roles", [])],
            travel_preference=value.get("travel_preference"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def evidence_map(self) -> dict[str, Evidence]:
        return {item.id: item for item in self.evidence}


@dataclass(frozen=True, slots=True)
class FormQuestion:
    label: str
    hint: str | None = None
    max_len: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FormQuestion":
        max_len = value.get("max_len")
        if max_len is not None and (not isinstance(max_len, int) or max_len <= 0):
            raise ValueError("max_len must be a positive integer")
        return cls(
            label=str(value["label"]),
            hint=value.get("hint"),
            max_len=max_len,
        )


@dataclass(frozen=True, slots=True)
class GroundedBlock:
    text: str
    kind: BlockKind
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GroundedBlock":
        return cls(
            text=str(value.get("text", "")).strip(),
            kind=BlockKind(value.get("kind", BlockKind.ANALYSIS)),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", [])),
        )


@dataclass(frozen=True, slots=True)
class ResponseSection:
    title: str
    blocks: tuple[GroundedBlock, ...]
    max_len: int | None = None
    compressed: bool = False

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text).strip()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResponseSection":
        max_len = value.get("max_len")
        return cls(
            title=str(value.get("title", "回答")),
            blocks=tuple(GroundedBlock.from_dict(item) for item in value.get("blocks", [])),
            max_len=int(max_len) if max_len is not None else None,
            compressed=bool(value.get("compressed", False)),
        )


@dataclass(frozen=True, slots=True)
class AgentRequest:
    workflow: Workflow
    style: Style
    payload: dict[str, Any]
    application_id: str | None = None
    plain_text: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentRequest":
        return cls(
            workflow=Workflow(value["workflow"]),
            style=Style(value.get("style", Style.TECHNICAL)),
            payload=dict(value.get("payload", {})),
            application_id=value.get("application_id"),
            plain_text=bool(value.get("plain_text", False)),
        )


@dataclass(frozen=True, slots=True)
class AgentResponse:
    workflow: Workflow
    style: Style
    sections: tuple[ResponseSection, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.value,
            "style": self.style.value,
            "sections": [
                {
                    "title": section.title,
                    "text": section.text,
                    "char_count": visible_char_count(section.text),
                    "max_len": section.max_len,
                    "compressed": section.compressed,
                    "evidence_ids": sorted(
                        {item for block in section.blocks for item in block.evidence_ids}
                    ),
                }
                for section in self.sections
            ],
            "warnings": list(self.warnings),
        }


def visible_char_count(text: str) -> int:
    """Count visible characters; whitespace used only for layout is ignored."""
    return sum(1 for char in text if not char.isspace())

