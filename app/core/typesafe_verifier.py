from __future__ import annotations

import os
from dataclasses import dataclass, field

from typesafe_sdk import Choice, Noul, TypeSafeClient


class TypeSafeEvaluationError(RuntimeError):
    pass


@dataclass(slots=True)
class StructuredVerification:
    score: float
    passed: bool
    feedback: str
    dimension_scores: dict[str, float] = field(default_factory=dict)
    choices: dict[str, str] = field(default_factory=dict)
    nouls: dict[str, float] = field(default_factory=dict)
    backend: str = "typesafe"


class TypeSafeVerifier:
    """JEV-style structured evaluator using TypeSafe system_one.

    The SDK calls the same style of endpoint shown in E:\\aicourse\\JEV\\test_jev.py:
    system_one(state=..., questions={...}) with Choice/Noul questions.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "jev-latest",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "").strip()
        self.base_url = base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.ergouzi.life").strip()
        self.model = model
        self.timeout = timeout

    @classmethod
    def available(cls, api_key: str | None = None) -> bool:
        key = api_key or os.environ.get("TYPESAFE_API_KEY", "").strip()
        return bool(key)

    def evaluate_sync(self, *, task: str, candidate: str, criteria: str) -> StructuredVerification:
        if not self.api_key:
            raise TypeSafeEvaluationError("TYPESAFE_API_KEY is not configured")

        questions = {
            "passed": Noul(
                instructions=(
                    "Does the candidate fully satisfy the task and criteria? "
                    "Answer true only when it is complete, correct, and ready to submit."
                ),
                criteria={
                    "true": "The candidate satisfies every material requirement.",
                    "false": "The candidate misses, contradicts, or weakly satisfies a requirement.",
                },
            ),
            "task_fit": Choice(
                instructions="How well does the candidate fit the task and criteria?",
                criteria={
                    "pass": "Fully satisfies the task and criteria.",
                    "partial": "Partially satisfies the task but has a concrete defect.",
                    "fail": "Does not satisfy the task or is unusable.",
                },
            ),
            "primary_issue": Choice(
                instructions="What is the primary remaining issue, if any?",
                criteria={
                    "none": "No material issue remains.",
                    "missing_requirement": "A stated requirement is missing.",
                    "format_violation": "The output format is wrong or inconsistent.",
                    "factual_risk": "There is a factual, logical, or hallucination risk.",
                    "unsafe": "The answer contains unsafe, destructive, or privacy-sensitive content.",
                },
            ),
            "hallucination_risk": Noul(
                instructions=(
                    "Does the candidate contain obviously unsupported, contradictory, or fabricated claims?"
                )
            ),
        }

        try:
            with TypeSafeClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            ) as client:
                response = client.system_one(
                    model=self.model,
                    state={
                        "task": task,
                        "criteria": criteria,
                        "candidate": candidate,
                    },
                    questions=questions,
                )
        except Exception as exc:
            raise TypeSafeEvaluationError(str(exc)) from exc

        passed_probability = float(response.nouls["passed"].noul)
        task_fit = response.choices["task_fit"]
        primary_issue = response.choices["primary_issue"]
        hallucination_risk = float(response.nouls["hallucination_risk"].noul)

        fit_score = {"pass": 1.0, "partial": 0.5, "fail": 0.0}.get(task_fit.choice, 0.0)
        hallucination_free = max(0.0, min(1.0, 1.0 - hallucination_risk))
        score = (0.5 * passed_probability) + (0.3 * fit_score) + (0.2 * hallucination_free)
        passed = (
            passed_probability >= 0.5
            and task_fit.choice != "fail"
            and primary_issue.choice != "unsafe"
            and hallucination_risk < 0.5
        )
        feedback = (
            f"TypeSafe judge: task_fit={task_fit.choice}, "
            f"primary_issue={primary_issue.choice}, "
            f"passed_probability={passed_probability:.3f}, "
            f"hallucination_risk={hallucination_risk:.3f}."
        )
        return StructuredVerification(
            score=max(0.0, min(1.0, score)),
            passed=passed,
            feedback=feedback,
            dimension_scores={
                "passed_probability": passed_probability,
                "task_fit": fit_score,
                "hallucination_free": hallucination_free,
            },
            choices={
                "task_fit": task_fit.choice,
                "primary_issue": primary_issue.choice,
            },
            nouls={
                "passed": passed_probability,
                "hallucination_risk": hallucination_risk,
            },
        )
