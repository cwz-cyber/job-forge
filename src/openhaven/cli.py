from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .agent import OpenHavenAgent
from .llm import OpenAICompatibleClient
from .models import AgentRequest, CandidateProfile, Workflow
from .policy import PolicyViolation
from .storage import JsonStateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenHaven 求职材料加工智能体")
    parser.add_argument("--profile", required=True, help="候选人素材 JSON")
    parser.add_argument("--request", required=True, help="任务请求 JSON")
    parser.add_argument("--state", help="可选的本地一致性与台账状态文件")
    parser.add_argument("--dry-run", action="store_true", help="只输出提示词请求包")
    parser.add_argument("--endpoint", help="Chat Completions compatible endpoint")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument(
        "--api-key-env",
        default="OPENHAVEN_API_KEY",
        help="保存 API key 的环境变量名",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = CandidateProfile.from_dict(_read_json(args.profile))
        request = AgentRequest.from_dict(_read_json(args.request))
        state = JsonStateStore(args.state) if args.state else None

        if args.dry_run:
            agent = OpenHavenAgent(profile=profile, state_store=state)
            _write_json(agent.prepare(request).to_dict())
            return 0

        if request.workflow == Workflow.APPLICATION_LEDGER:
            agent = OpenHavenAgent(profile=profile, state_store=state)
            _write_json(agent.run(request).to_dict())
            return 0

        if not args.endpoint or not args.model:
            raise ValueError("非 dry-run 模式必须提供 --endpoint 和 --model")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {args.api_key_env} 未设置")
        llm = OpenAICompatibleClient(
            endpoint=args.endpoint,
            api_key=api_key,
            model=args.model,
        )
        agent = OpenHavenAgent(profile=profile, llm=llm, state_store=state)
        _write_json(agent.run(request).to_dict())
        return 0
    except PolicyViolation as exc:
        _write_error(
            "policy_violation",
            [
                {"code": item.code, "message": item.message, "section": item.section}
                for item in exc.issues
            ],
        )
        return 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        _write_error("request_failed", str(exc))
        return 1


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是 JSON object")
    return value


def _write_json(value: dict[str, Any]) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _write_error(code: str, detail: Any) -> None:
    json.dump({"error": code, "detail": detail}, sys.stderr, ensure_ascii=False, indent=2)
    sys.stderr.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

