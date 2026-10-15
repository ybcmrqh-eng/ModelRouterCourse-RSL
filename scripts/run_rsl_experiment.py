#!/usr/bin/env python3
"""Run reproducible bounded-RSL experiments against a running ModelRouter."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def experiment_requests(model: str) -> list[dict[str, Any]]:
    common = {
        "model": model,
        "evaluator_backend": "auto",
        "max_rounds": 2,
        "candidates_per_round": 3,
        "top_k": 2,
        "patience": 1,
        "target_score": 0.9,
        "temperature": 0.6,
        "diversity_threshold": 0.35,
        "max_tokens": 320,
        "seed": 20261015,
        "memory_min_evidence": 2,
        "memory_min_score": 0.9,
        "budget": {
            "max_model_calls": 24,
            "max_total_tokens": 16000,
            "max_duration_seconds": 300,
        },
    }
    return [
        {
            **common,
            "task": "输出一个 JSON 对象，键为 name、age、skills；skills 必须是恰好 3 个字符串；不得输出 Markdown 代码围栏或解释。",
            "criteria": "输出必须能被 JSON 解析；必须包含 name、age、skills；skills 必须恰好三个字符串；不能有代码围栏或额外解释。",
            "constraints": [
                {"name": "no markdown fence", "kind": "not_contains", "value": "```"},
                {"name": "contains name key", "kind": "contains", "value": "name"},
                {"name": "contains age key", "kind": "contains", "value": "age"},
                {"name": "contains skills key", "kind": "contains", "value": "skills"},
            ],
            "external_checks": [
                {"name": "json valid", "kind": "json_valid"},
                {"name": "required keys", "kind": "contains_all", "values": ["name", "age", "skills"]},
                {"name": "three skill strings", "kind": "regex", "value": r'"skills"\s*:\s*\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'},
                {"name": "json only", "kind": "regex", "value": r'^\s*\{[\s\S]*\}\s*$'},
            ],
            "holdout_checks": [
                {"name": "holdout three skills", "kind": "regex", "value": r'"skills"\s*:\s*\[[^\]]+\]'},
            ],
        },
        {
            **common,
            "task": "只输出一条 SQLite 查询：从 users 表读取 age 大于参数 :min_age 的 name，并按 name 升序排序。不得使用字面数字 18，不得输出 Markdown 围栏或解释。",
            "criteria": "必须是可执行的 SQLite SELECT 查询；使用 :min_age 参数；包含 WHERE 和 ORDER BY name；不能出现字面数字 18；不能有 Markdown 围栏。",
            "constraints": [
                {"name": "no markdown fence", "kind": "not_contains", "value": "```"},
                {"name": "uses parameter", "kind": "contains", "value": ":min_age"},
                {"name": "no literal 18", "kind": "not_contains", "value": "18"},
            ],
            "external_checks": [
                {"name": "select name from users", "kind": "regex", "value": r"(?i)select\s+name\s+from\s+users"},
                {"name": "parameter present", "kind": "contains", "value": ":min_age"},
                {"name": "where present", "kind": "regex", "value": r"(?i)\bwhere\b"},
                {"name": "order by name", "kind": "regex", "value": r"(?i)order\s+by\s+name"},
                {"name": "no literal 18", "kind": "not_contains", "value": "18"},
            ],
            "holdout_checks": [
                {"name": "holdout select", "kind": "regex", "value": r"(?i)select.+from.+where"},
            ],
        },
        {
            **common,
            "task": "只输出 Python 代码：实现 safe_divide(a: float, b: float) -> float | None；b 为 0 时返回 None；包含 docstring；不得输出 Markdown 围栏或解释。",
            "criteria": "代码必须通过 Python 语法检查；包含 def safe_divide；包含类型注解；包含 docstring；b 为 0 时返回 None；不能有 Markdown 围栏。",
            "constraints": [
                {"name": "no markdown fence", "kind": "not_contains", "value": "```"},
                {"name": "def present", "kind": "contains", "value": "def safe_divide"},
                {"name": "docstring present", "kind": "contains", "value": '"""'},
            ],
            "external_checks": [
                {"name": "python syntax", "kind": "python_syntax"},
                {"name": "def present", "kind": "contains", "value": "def safe_divide"},
                {"name": "type annotations", "kind": "regex", "value": r"def\s+safe_divide\s*\([^)]*:\s*(?:float|int)[^)]*\)\s*->"},
                {"name": "zero returns None", "kind": "regex", "value": r"(?s)if\s+[^:]+==\s*0\s*:\s*\n\s*return\s+None"},
            ],
            "holdout_checks": [
                {"name": "holdout returns", "kind": "regex", "value": r"(?s)return\s+"},
            ],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-router-key")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--output-dir", default=r"docs\evidence")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {args.api_key}"}

    with httpx.Client(timeout=600) as client:
        for index, request in enumerate(experiment_requests(args.model), start=1):
            response = client.post(f"{args.base_url.rstrip('/')}/v1/rsl/run", headers=headers, json=request)
            response.raise_for_status()
            result = response.json()
            results.append({"experiment": index, "task": request["task"], "result": result})
            print(
                f"experiment={index} "
                f"baseline={result['baseline_score']:.4f} "
                f"best={result['best_score']:.4f} "
                f"verified_improvement={result['verified_improvement']} "
                f"stopped={result['stopped_reason']} "
                f"rounds={len(result['rounds'])}"
            )

    output = out_dir / f"rsl-experiment-{run_id}.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
