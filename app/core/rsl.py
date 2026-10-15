from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.db import Database
from app.core.request_context import RequestContext
from app.core.router import RouterEngine
from app.core.typesafe_verifier import TypeSafeEvaluationError, TypeSafeVerifier
from app.schemas.chat import ChatMessage, ChatRequest
from app.schemas.rsl import (
    RSLCandidate,
    RSLConstraint,
    RSLEvaluation,
    RSLExternalCheck,
    RSLMemoryItem,
    RSLRound,
    RSLRunBudget,
    RSLRunMetrics,
    RSLRunRequest,
    RSLRunResponse,
    RSLRunSummary,
)

logger = logging.getLogger("model_router.rsl")

PROMPT_VERSION = "rsl-prompts-v2"
VERIFIER_VERSION = "rsl-verifier-v2"
MEMORY_SCHEMA_VERSION = 2

GENERATOR_SYSTEM_PROMPT = """You are the candidate generator in a bounded Recursive Self-Learning system.
Produce the best possible answer to the task. Use the supplied reusable lessons and evaluator feedback when available.
Do not explain the learning process. Return only the final answer for the requested task."""

EVALUATOR_SYSTEM_PROMPT = """You are an independent evaluator for a Recursive Self-Learning system.
Evaluate the candidate against the original task and criteria. Do not assume the candidate is correct.
Return one JSON object only, with no markdown fences or extra text.
Required keys: score, passed, feedback, strengths, weaknesses, dimension_scores.
score and each dimension value must be a real number from 0 to 1. passed must be a real boolean.
feedback must be specific and actionable, naming the missing field, wrong constraint, unsupported claim, or format defect.
A fully compliant answer normally scores at least 0.85. Use 0 only for irrelevant, unsafe, or unusable answers."""

SAFETY_PATTERNS = {
    "destructive_command": r"\brm\s+-rf\b|\bformat\s+[a-z]:",
    "shell_execution": r"\bos\.system\s*\(|subprocess\.(?:run|Popen)\s*\([^)]*shell\s*=\s*True",
    "credential_literal": r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*['\"][^'\"]{8,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _similarity_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    latin = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]", normalized)
    return latin | {cjk[i] + cjk[i + 1] for i in range(max(0, len(cjk) - 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:16]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denominator_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    denominator = denominator_left * denominator_right
    if denominator == 0:
        return None
    return max(-1.0, min(1.0, numerator / denominator))


@dataclass
class _CandidateCall:
    answer: str
    raw_answer: str
    postprocessed: bool
    usage: dict | None
    latency_ms: int
    seed: int | None


class _BudgetManager:
    def __init__(self, budget: RSLRunBudget):
        self.budget = budget
        self.started = time.monotonic()
        self.model_calls = 0
        self.generation_calls = 0
        self.evaluator_calls = 0
        self.total_tokens = 0

    def stop_reason(self) -> str | None:
        if self.model_calls >= self.budget.max_model_calls:
            return "budget_model_calls"
        if self.total_tokens >= self.budget.max_total_tokens:
            return "budget_total_tokens"
        if time.monotonic() - self.started >= self.budget.max_duration_seconds:
            return "budget_duration"
        return None

    def record(self, *, usage: dict | None, kind: str) -> None:
        self.model_calls += 1
        if kind == "generation":
            self.generation_calls += 1
        else:
            self.evaluator_calls += 1
        self.total_tokens += int((usage or {}).get("total_tokens") or 0)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "model_calls": self.model_calls,
            "generation_calls": self.generation_calls,
            "evaluator_calls": self.evaluator_calls,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
        }


class RSLEngine:
    """Bounded recursive self-learning on top of ModelRouter."""

    def __init__(self, db: Database, router: RouterEngine):
        self.db = db
        self.router = router
        self.typesafe = TypeSafeVerifier()

    async def initialize(self):
        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                subject_id TEXT DEFAULT '',
                model TEXT NOT NULL,
                evaluator_model TEXT NOT NULL,
                task TEXT NOT NULL,
                task_signature TEXT NOT NULL,
                status TEXT NOT NULL,
                current_round INTEGER DEFAULT 0,
                target_score REAL DEFAULT 0.85,
                initial_score REAL DEFAULT 0,
                baseline_score REAL DEFAULT 0,
                best_score REAL DEFAULT 0,
                improvement REAL DEFAULT 0,
                verified_improvement REAL,
                best_round INTEGER DEFAULT 0,
                learned_strategy TEXT DEFAULT '',
                memory_used INTEGER DEFAULT 0,
                memory_written INTEGER DEFAULT 0,
                memory_id INTEGER,
                stopped_reason TEXT DEFAULT '',
                verification_mode TEXT DEFAULT 'internal_only',
                verifier_version TEXT DEFAULT '',
                prompt_version TEXT DEFAULT '',
                candidates_per_round INTEGER DEFAULT 3,
                top_k INTEGER DEFAULT 2,
                controls_json TEXT DEFAULT '{}',
                metrics_json TEXT DEFAULT '{}',
                budget_json TEXT DEFAULT '{}',
                duration_ms INTEGER DEFAULT 0,
                error TEXT
            )
            """
        )
        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES rsl_runs(run_id) ON DELETE CASCADE,
                round_index INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                answer TEXT NOT NULL,
                score REAL NOT NULL,
                passed INTEGER DEFAULT 0,
                feedback TEXT DEFAULT '',
                strengths_json TEXT DEFAULT '[]',
                weaknesses_json TEXT DEFAULT '[]',
                dimensions_json TEXT DEFAULT '{}',
                usage_json TEXT DEFAULT '{}',
                latency_ms INTEGER DEFAULT 0,
                accepted INTEGER DEFAULT 0,
                memory_hits_json TEXT DEFAULT '[]',
                candidates_json TEXT DEFAULT '[]',
                average_score REAL DEFAULT 0,
                score_std REAL DEFAULT 0,
                diversity REAL DEFAULT 0,
                duplicate_rate REAL DEFAULT 0,
                best_candidate_index INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES rsl_runs(run_id) ON DELETE CASCADE,
                round_index INTEGER NOT NULL,
                candidate_index INTEGER NOT NULL,
                answer TEXT NOT NULL,
                raw_answer TEXT,
                postprocessed INTEGER DEFAULT 0,
                content_hash TEXT NOT NULL,
                score REAL NOT NULL,
                passed INTEGER DEFAULT 0,
                evaluator_score REAL DEFAULT 0,
                format_score REAL,
                external_score REAL,
                process_score REAL DEFAULT 0,
                feedback TEXT DEFAULT '',
                safety_flags_json TEXT DEFAULT '[]',
                evidence_json TEXT DEFAULT '[]',
                dimensions_json TEXT DEFAULT '{}',
                usage_json TEXT DEFAULT '{}',
                latency_ms INTEGER DEFAULT 0,
                accepted INTEGER DEFAULT 0,
                is_best INTEGER DEFAULT 0,
                seed INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                task_signature TEXT NOT NULL,
                task_example TEXT NOT NULL,
                lesson TEXT NOT NULL,
                score_gain REAL DEFAULT 0,
                score_before REAL,
                score_after REAL,
                source_run_id TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'candidate',
                evidence_count INTEGER DEFAULT 1,
                previous_version_id INTEGER,
                rollback_reason TEXT,
                rolled_back_at TEXT
            )
            """
        )

        for table, column, ddl in [
            ("rsl_runs", "baseline_score", "baseline_score REAL DEFAULT 0"),
            ("rsl_runs", "verified_improvement", "verified_improvement REAL"),
            ("rsl_runs", "memory_written", "memory_written INTEGER DEFAULT 0"),
            ("rsl_runs", "memory_id", "memory_id INTEGER"),
            ("rsl_runs", "stopped_reason", "stopped_reason TEXT DEFAULT ''"),
            ("rsl_runs", "verification_mode", "verification_mode TEXT DEFAULT 'internal_only'"),
            ("rsl_runs", "verifier_version", "verifier_version TEXT DEFAULT ''"),
            ("rsl_runs", "prompt_version", "prompt_version TEXT DEFAULT ''"),
            ("rsl_runs", "candidates_per_round", "candidates_per_round INTEGER DEFAULT 3"),
            ("rsl_runs", "top_k", "top_k INTEGER DEFAULT 2"),
            ("rsl_runs", "controls_json", "controls_json TEXT DEFAULT '{}'"),
            ("rsl_runs", "metrics_json", "metrics_json TEXT DEFAULT '{}'"),
            ("rsl_runs", "budget_json", "budget_json TEXT DEFAULT '{}'"),
            ("rsl_candidates", "raw_answer", "raw_answer TEXT"),
            ("rsl_candidates", "postprocessed", "postprocessed INTEGER DEFAULT 0"),
            ("rsl_rounds", "candidates_json", "candidates_json TEXT DEFAULT '[]'"),
            ("rsl_rounds", "average_score", "average_score REAL DEFAULT 0"),
            ("rsl_rounds", "score_std", "score_std REAL DEFAULT 0"),
            ("rsl_rounds", "diversity", "diversity REAL DEFAULT 0"),
            ("rsl_rounds", "duplicate_rate", "duplicate_rate REAL DEFAULT 0"),
            ("rsl_rounds", "best_candidate_index", "best_candidate_index INTEGER DEFAULT 0"),
            ("rsl_memories", "score_before", "score_before REAL"),
            ("rsl_memories", "score_after", "score_after REAL"),
            ("rsl_memories", "version", "version INTEGER DEFAULT 1"),
            ("rsl_memories", "status", "status TEXT DEFAULT 'candidate'"),
            ("rsl_memories", "evidence_count", "evidence_count INTEGER DEFAULT 1"),
            ("rsl_memories", "previous_version_id", "previous_version_id INTEGER"),
            ("rsl_memories", "rollback_reason", "rollback_reason TEXT"),
            ("rsl_memories", "rolled_back_at", "rolled_back_at TEXT"),
        ]:
            await self.db._ensure_column(table, column, ddl)

        await self.db._execute(
            "UPDATE rsl_memories SET status='legacy' WHERE status IS NULL OR status=''"
        )
        await self.db._execute(
            "UPDATE rsl_memories SET status='legacy' "
            "WHERE status='candidate' AND score_before IS NULL AND score_after IS NULL"
        )
        await self.db._execute(
            "CREATE INDEX IF NOT EXISTS idx_rsl_candidates_run ON rsl_candidates(run_id, round_index, candidate_index)"
        )
        await self.db._execute(
            "CREATE INDEX IF NOT EXISTS idx_rsl_memories_active ON rsl_memories(model, task_signature, status, version DESC)"
        )
        await self.db._conn.commit()

    async def run(
        self,
        request: RSLRunRequest,
        request_context: RequestContext | None = None,
    ) -> RSLRunResponse:
        run_id = f"rsl-{uuid.uuid4().hex[:16]}"
        created_at = _utc_now()
        started = time.monotonic()
        evaluator_model = request.evaluator_model or request.model
        task_signature = _normalize_text(request.task)
        subject_id = request_context.subject_id if request_context else ""
        budget = _BudgetManager(request.budget)

        if request.persist:
            await self.db._execute(
                """
                INSERT INTO rsl_runs
                (run_id,created_at,subject_id,model,evaluator_model,task,task_signature,status,
                 current_round,target_score,memory_used,candidates_per_round,top_k,
                 verifier_version,prompt_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    created_at,
                    subject_id,
                    request.model,
                    evaluator_model,
                    request.task,
                    task_signature,
                    "running",
                    0,
                    request.target_score,
                    0,
                    request.candidates_per_round,
                    request.top_k,
                    VERIFIER_VERSION,
                    PROMPT_VERSION,
                ),
            )
            await self.db._conn.commit()

        try:
            memories = (
                await self._retrieve_memories(
                    task=request.task,
                    model=request.model,
                    limit=request.memory_limit,
                )
                if request.use_memory and request.memory_limit > 0
                else []
            )

            rounds: list[RSLRound] = []
            all_candidates: list[RSLCandidate] = []
            best_answer = ""
            best_round = 0
            best_score = -1.0
            baseline_score = 0.0
            control_single = 0.0
            control_revise = 0.0
            last_round_diversity = 1.0
            no_improvement_rounds = 0
            stopped_reason = "max_rounds"

            if request.baseline_enabled:
                baseline_call = await self._generate_candidate(
                    request=request,
                    task=request.task,
                    criteria=request.criteria,
                    memories=memories,
                    parent_answer="",
                    parent_evaluation=None,
                    round_index=0,
                    candidate_index=0,
                    diversity_hint="",
                    budget=budget,
                )
                baseline_eval = await self._evaluate_candidate(
                    request=request,
                    task=request.task,
                    answer=baseline_call.answer,
                    criteria=request.criteria,
                    evaluator_model=evaluator_model,
                    budget=budget,
                )
                baseline_candidate = RSLCandidate(
                    candidate_index=0,
                    answer=baseline_call.answer,
                    evaluation=baseline_eval,
                    usage=baseline_call.usage,
                    latency_ms=baseline_call.latency_ms,
                    accepted=True,
                    is_best=True,
                    content_hash=_content_hash(baseline_call.answer),
                    seed=baseline_call.seed,
                )
                baseline_round = self._make_round(
                    round_index=0,
                    strategy="baseline_single_generation",
                    candidates=[baseline_candidate],
                    memory_hits=[item.lesson for item in memories],
                )
                rounds.append(baseline_round)
                all_candidates.append(baseline_candidate)
                baseline_score = baseline_eval.score
                best_score = baseline_eval.score
                best_answer = baseline_call.answer
                best_round = 0
                control_single = baseline_eval.score
                control_revise = baseline_eval.score
                if request.persist:
                    await self._persist_round(run_id, baseline_round)

            pool = list(all_candidates)
            stop_now = budget.stop_reason()
            if stop_now:
                stopped_reason = stop_now
            elif not request.baseline_enabled:
                stopped_reason = "baseline_disabled_no_search_parent"
            elif baseline_score >= request.target_score and rounds[0].evaluation.passed:
                stopped_reason = "target_reached_at_baseline"
            else:
                for round_index in range(1, request.max_rounds + 1):
                    stop_now = budget.stop_reason()
                    if stop_now:
                        stopped_reason = stop_now
                        break

                    diversity_hint = ""
                    if last_round_diversity < request.diversity_threshold:
                        diversity_hint = (
                            "The previous candidate pool was structurally similar. "
                            "Use a materially different strategy, structure, or wording while satisfying the task."
                        )

                    candidates: list[RSLCandidate] = []
                    for candidate_index in range(request.candidates_per_round):
                        if budget.stop_reason():
                            break
                        parent = pool[candidate_index % len(pool)] if pool else None
                        call = await self._generate_candidate(
                            request=request,
                            task=request.task,
                            criteria=request.criteria,
                            memories=memories,
                            parent_answer=parent.answer if parent else best_answer,
                            parent_evaluation=parent.evaluation if parent else None,
                            round_index=round_index,
                            candidate_index=candidate_index,
                            diversity_hint=diversity_hint,
                            budget=budget,
                        )
                        evaluation = await self._evaluate_candidate(
                            request=request,
                            task=request.task,
                            answer=call.answer,
                            criteria=request.criteria,
                            evaluator_model=evaluator_model,
                            budget=budget,
                        )
                        candidates.append(
                            RSLCandidate(
                                candidate_index=candidate_index,
                                answer=call.answer,
                                evaluation=evaluation,
                                usage=call.usage,
                                latency_ms=call.latency_ms,
                                content_hash=_content_hash(call.answer),
                                seed=call.seed,
                            )
                        )

                    if not candidates:
                        stopped_reason = budget.stop_reason() or "no_candidates_generated"
                        break

                    candidates.sort(key=lambda item: item.evaluation.score, reverse=True)
                    for rank, candidate in enumerate(candidates):
                        candidate.is_best = rank == 0
                        candidate.accepted = candidate.evaluation.score > best_score

                    previous_best = best_score
                    for candidate in candidates:
                        all_candidates.append(candidate)
                        if candidate.evaluation.score > best_score:
                            best_score = candidate.evaluation.score
                            best_answer = candidate.answer
                            best_round = round_index

                    first_candidate = next(
                        (item for item in candidates if item.candidate_index == 0),
                        None,
                    )
                    if first_candidate and first_candidate.evaluation.score > control_revise:
                        control_revise = first_candidate.evaluation.score

                    round_result = self._make_round(
                        round_index=round_index,
                        strategy="candidate_search_topk",
                        candidates=candidates,
                        memory_hits=[item.lesson for item in memories],
                    )
                    rounds.append(round_result)
                    if request.persist:
                        await self._persist_round(run_id, round_result)

                    last_round_diversity = round_result.diversity
                    pool = sorted(
                        pool + candidates,
                        key=lambda item: item.evaluation.score,
                        reverse=True,
                    )[: request.top_k]
                    if not pool:
                        stopped_reason = "empty_candidate_pool"
                        break

                    improvement = best_score - previous_best
                    no_improvement_rounds = no_improvement_rounds + 1 if improvement <= 1e-9 else 0
                    best_candidate = next(
                        (item for item in all_candidates if item.answer == best_answer),
                        candidates[0],
                    )
                    if best_score >= request.target_score and best_candidate.evaluation.passed:
                        stopped_reason = "target_reached"
                        break
                    if no_improvement_rounds >= request.patience:
                        stopped_reason = "no_improvement"
                        break
                    if budget.stop_reason():
                        stopped_reason = budget.stop_reason() or "budget"
                        break

            if not best_answer:
                raise RuntimeError("RSL finished without a generated candidate")

            holdout_score = None
            if request.holdout_checks:
                holdout_score, evidence = self._evaluate_external_checks(
                    best_answer,
                    request.holdout_checks,
                )
                best_candidate = next(
                    (item for item in all_candidates if item.answer == best_answer),
                    all_candidates[0],
                )
                if holdout_score < 1.0:
                    best_candidate.evaluation.passed = False
                    best_candidate.evaluation.weaknesses.extend(
                        f"Holdout check failed: {item}" for item in evidence if item.startswith("FAIL")
                    )

            best_evaluation = next(
                (candidate.evaluation for candidate in all_candidates if candidate.answer == best_answer),
                all_candidates[0].evaluation,
            )
            learned_strategy = self._build_learned_strategy(
                best_evaluation=best_evaluation,
                best_round=best_round,
                improvement=max(0.0, best_score - baseline_score),
            )

            memory_written = False
            memory_id = None
            if request.memory_write_enabled and request.persist:
                memory_result = await self._store_verified_memory(
                    request=request,
                    task_signature=task_signature,
                    lesson=learned_strategy,
                    score_before=baseline_score,
                    score_after=best_score,
                    source_run_id=run_id,
                    best_candidate=next(
                        (candidate for candidate in all_candidates if candidate.answer == best_answer),
                        all_candidates[0],
                    ),
                    holdout_score=holdout_score,
                )
                if memory_result:
                    memory_id, memory_written = memory_result

            final_candidate = next(
                (candidate for candidate in all_candidates if candidate.answer == best_answer),
                all_candidates[0],
            )
            verified_pass = (
                final_candidate.evaluation.external_score is not None
                and final_candidate.evaluation.external_score == 1.0
                and (final_candidate.evaluation.format_score is None or final_candidate.evaluation.format_score == 1.0)
                and not final_candidate.evaluation.safety_flags
                and (holdout_score is None or holdout_score == 1.0)
            )
            metrics = self._build_metrics(
                request=request,
                baseline_score=baseline_score,
                best_score=best_score,
                all_candidates=all_candidates,
                rounds=rounds,
                holdout_score=holdout_score,
                budget=budget,
                verified_pass=verified_pass,
            )
            controls = {
                "single_generation": round(control_single, 6),
                "generate_evaluate_revise": round(control_revise, 6),
                "candidate_search_topk": round(best_score, 6),
            }
            verification_mode = self._verification_mode(request)
            duration_ms = int((time.monotonic() - started) * 1000)
            completed_at = _utc_now()
            verified_improvement = max(0.0, best_score - baseline_score) if verified_pass else None

            if request.persist:
                await self._finish_run(
                    run_id=run_id,
                    status="completed",
                    completed_at=completed_at,
                    current_round=len(rounds) - 1,
                    baseline_score=baseline_score,
                    best_score=best_score,
                    improvement=max(0.0, best_score - baseline_score),
                    verified_improvement=verified_improvement,
                    best_round=best_round,
                    learned_strategy=learned_strategy,
                    memory_used=len(memories),
                    memory_written=memory_written,
                    memory_id=memory_id,
                    stopped_reason=stopped_reason,
                    verification_mode=verification_mode,
                    metrics=metrics,
                    controls=controls,
                    budget_used=budget.snapshot(),
                    duration_ms=duration_ms,
                )

            return RSLRunResponse(
                run_id=run_id,
                created_at=created_at,
                completed_at=completed_at,
                task=request.task,
                model=request.model,
                evaluator_model=evaluator_model,
                status="completed",
                baseline_score=baseline_score,
                initial_score=baseline_score,
                best_score=best_score,
                improvement=max(0.0, best_score - baseline_score),
                verified_improvement=verified_improvement,
                verification_mode=verification_mode,
                best_round=best_round,
                best_answer=best_answer,
                rounds=rounds,
                learned_strategy=learned_strategy,
                memory_used=len(memories),
                memory_written=memory_written,
                memory_id=memory_id,
                stopped_reason=stopped_reason,
                verifier_version=VERIFIER_VERSION,
                prompt_version=PROMPT_VERSION,
                controls=controls,
                metrics=metrics,
                budget_used=budget.snapshot(),
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            if request.persist:
                await self.db._execute(
                    """
                    UPDATE rsl_runs
                    SET completed_at=?,status='failed',duration_ms=?,error=?
                    WHERE run_id=?
                    """,
                    (_utc_now(), duration_ms, str(exc)[:1000], run_id),
                )
                await self.db._conn.commit()
            logger.exception("RSL run %s failed", run_id)
            raise

    async def _generate_candidate(
        self,
        *,
        request: RSLRunRequest,
        task: str,
        criteria: str,
        memories: list[RSLMemoryItem],
        parent_answer: str,
        parent_evaluation: RSLEvaluation | None,
        round_index: int,
        candidate_index: int,
        diversity_hint: str,
        budget: _BudgetManager,
    ) -> _CandidateCall:
        stop_reason = budget.stop_reason()
        if stop_reason:
            raise RuntimeError(f"RSL budget exhausted before generation: {stop_reason}")

        sections = [f"TASK:\n{task}", f"SUCCESS CRITERIA:\n{criteria}"]
        output_contract = self._format_output_contract(request)
        if output_contract:
            sections.append(f"OUTPUT CONTRACT (must pass):\n{output_contract}")
        if memories:
            lessons = "\n".join(f"- {item.lesson}" for item in memories)
            sections.append(f"REUSABLE VERIFIED LESSONS:\n{lessons}")
        if round_index == 0:
            sections.append("This is the baseline candidate. Produce one complete, directly checkable answer.")
        else:
            if parent_answer:
                sections.append(f"PARENT CANDIDATE:\n{parent_answer}")
            if parent_evaluation:
                weaknesses = "\n".join(f"- {item}" for item in parent_evaluation.weaknesses)
                weaknesses = weaknesses or "- No explicit weakness was returned."
                sections.append(
                    "INDEPENDENT EVALUATOR FEEDBACK:\n"
                    f"{parent_evaluation.feedback}\n"
                    f"WEAKNESSES TO FIX:\n{weaknesses}"
                )
                failed_checks = [item for item in parent_evaluation.evidence if item.startswith("FAIL ")]
                if failed_checks:
                    sections.append(
                        "UNRESOLVED INDEPENDENT CHECK FAILURES (fix every item):\n"
                        + "\n".join(f"- {item}" for item in failed_checks)
                    )
            if diversity_hint:
                sections.append(f"DIVERSITY REQUIREMENT:\n{diversity_hint}")
            sections.append(
                f"This is candidate variant {candidate_index + 1}. "
                "Return the complete revised answer only. Do not explain the revision."
            )

        messages = [
            ChatMessage(role="system", content=GENERATOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n\n".join(sections)),
        ]
        temperature = min(request.temperature + (candidate_index * 0.15) + (round_index * 0.05), 2.0)
        seed = request.seed + (round_index * 100) + candidate_index if request.seed is not None else None
        raw_answer, usage, latency_ms = await self._call_model(
            model=request.model,
            messages=messages,
            temperature=temperature,
            max_tokens=request.max_tokens,
            seed=seed,
        )
        answer, postprocessed = self._sanitize_candidate(raw_answer, request)
        budget.record(usage=usage, kind="generation")
        return _CandidateCall(
            answer=answer,
            raw_answer=raw_answer,
            postprocessed=postprocessed,
            usage=usage,
            latency_ms=latency_ms,
            seed=seed,
        )

    async def _evaluate_candidate(
        self,
        *,
        request: RSLRunRequest,
        task: str,
        answer: str,
        criteria: str,
        evaluator_model: str,
        budget: _BudgetManager,
    ) -> RSLEvaluation:
        stop_reason = budget.stop_reason()
        if stop_reason:
            raise RuntimeError(f"RSL budget exhausted before evaluation: {stop_reason}")

        local_evaluation = await self._evaluate_local(
            task=task,
            answer=answer,
            criteria=criteria,
            evaluator_model=evaluator_model,
            budget=budget,
        )
        process_scores = [local_evaluation.score]
        backends = ["local"]
        feedback_parts = [local_evaluation.feedback]
        dimensions = dict(local_evaluation.dimension_scores)
        strengths = list(local_evaluation.strengths)
        weaknesses = list(local_evaluation.weaknesses)

        if request.evaluator_backend in {"auto", "typesafe"} and TypeSafeVerifier.available():
            try:
                structured = await asyncio.to_thread(
                    self.typesafe.evaluate_sync,
                    task=task,
                    candidate=answer,
                    criteria=criteria,
                )
                budget.record(usage=None, kind="evaluation")
                process_scores.append(structured.score)
                backends.append(structured.backend)
                feedback_parts.append(structured.feedback)
                dimensions.update(
                    {f"typesafe_{key}": value for key, value in structured.dimension_scores.items()}
                )
                if structured.choices.get("primary_issue") not in {None, "none"}:
                    weaknesses.append(f"TypeSafe primary issue: {structured.choices['primary_issue']}")
                if structured.passed:
                    strengths.append("TypeSafe structured evaluator marked the candidate as passed.")
            except TypeSafeEvaluationError as exc:
                logger.warning("TypeSafe evaluation unavailable; using local evaluator: %s", exc)

        format_score, format_evidence = self._evaluate_constraints(answer, request.constraints)
        external_score, external_evidence = self._evaluate_external_checks(answer, request.external_checks)
        safety_flags = self._safety_flags(answer)
        if safety_flags:
            external_score = 0.0
            external_evidence.extend(f"FAIL safety:{flag}" for flag in safety_flags)

        if local_evaluation.evaluator_backends == ["local_fallback"] and len(process_scores) == 1:
            if external_score == 1.0 and (format_score is None or format_score == 1.0) and not safety_flags:
                process_scores = [1.0]
            elif external_score == 0.0 or format_score == 0.0 or safety_flags:
                process_scores = [0.0]
            else:
                process_scores = [0.5]
        process_score = max(0.0, min(1.0, statistics.fmean(process_scores)))

        if external_score is not None and format_score is not None:
            combined = (0.6 * external_score) + (0.2 * format_score) + (0.2 * process_score)
        elif external_score is not None:
            combined = (0.8 * external_score) + (0.2 * process_score)
        elif format_score is not None:
            combined = (0.6 * format_score) + (0.4 * process_score)
        else:
            combined = process_score
        combined = max(0.0, min(1.0, combined))

        passed = (
            not safety_flags
            and (external_score is None or external_score == 1.0)
            and (format_score is None or format_score == 1.0)
            and process_score >= 0.8
        )
        feedback = " ".join(part for part in feedback_parts if part).strip()
        if external_evidence:
            feedback = f"{feedback} Independent checks: {'; '.join(external_evidence)}".strip()
        if format_evidence:
            feedback = f"{feedback} Format checks: {'; '.join(format_evidence)}".strip()

        return RSLEvaluation(
            score=combined,
            passed=passed,
            feedback=feedback,
            strengths=strengths,
            weaknesses=weaknesses,
            dimension_scores=dimensions,
            evaluator_score=process_score,
            format_score=format_score,
            external_score=external_score,
            process_score=process_score,
            evidence=[*external_evidence, *format_evidence],
            safety_flags=safety_flags,
            evaluator_backends=backends,
        )

    async def _evaluate_local(
        self,
        *,
        task: str,
        answer: str,
        criteria: str,
        evaluator_model: str,
        budget: _BudgetManager,
    ) -> RSLEvaluation:
        prompt = (
            f"ORIGINAL TASK:\n{task}\n\n"
            f"SUCCESS CRITERIA:\n{criteria}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            "Evaluate only from the information above. Name concrete missing fields or format defects."
        )
        messages = [
            ChatMessage(role="system", content=EVALUATOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
        content, usage, _ = await self._call_model(
            model=evaluator_model,
            messages=messages,
            temperature=0.0,
            max_tokens=450,
            seed=None,
        )
        budget.record(usage=usage, kind="evaluation")
        try:
            return self._parse_evaluation(content)
        except Exception as exc:
            logger.warning("RSL local evaluator returned invalid structured output: %s", exc)
            return self._fallback_evaluation(answer, str(exc))

    async def _call_model(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> tuple[str, dict | None, int]:
        provider, model_config = await self.router.route(model)
        request = ChatRequest(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        started = time.monotonic()
        response = await provider.chat_completions(request, model_config)
        latency_ms = int((time.monotonic() - started) * 1000)
        if not response.choices:
            raise RuntimeError(f"Model {model} returned no choices")
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError(f"Model {model} returned an empty answer")
        usage = response.usage.model_dump() if response.usage else None
        return content, usage, latency_ms

    def _parse_evaluation(self, content: str) -> RSLEvaluation:
        payload = json.loads(self._extract_json_object(content))
        if self._looks_like_placeholder(payload):
            raise ValueError("evaluator returned template placeholder values")
        score = self._parse_score(payload.get("score"))
        if score is None:
            raise ValueError("evaluation JSON is missing a numeric score")
        passed = self._parse_bool(payload.get("passed"))
        if passed is None:
            passed = score >= 0.85
        dimensions = payload.get("dimension_scores") or {}
        if not isinstance(dimensions, dict):
            dimensions = {}
        cleaned_dimensions = {}
        for key, value in dimensions.items():
            parsed = self._parse_score(value)
            if parsed is not None:
                cleaned_dimensions[str(key)] = parsed
        return RSLEvaluation(
            score=score,
            passed=passed,
            feedback=str(payload.get("feedback") or "No evaluator feedback.").strip(),
            strengths=self._parse_string_list(payload.get("strengths")),
            weaknesses=self._parse_string_list(payload.get("weaknesses")),
            dimension_scores=cleaned_dimensions,
            evaluator_score=score,
            process_score=score,
            evaluator_backends=["local"],
        )

    def _extract_json_object(self, content: str) -> str:
        stripped = (content or "").strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if match:
                return match.group(0)
        raise ValueError("evaluator did not return a JSON object")

    def _looks_like_placeholder(self, payload: dict) -> bool:
        feedback = str(payload.get("feedback") or "").strip().lower()
        if feedback in {"short concrete feedback", "specific sentence", "..."}:
            return True
        strengths = self._parse_string_list(payload.get("strengths"))
        weaknesses = self._parse_string_list(payload.get("weaknesses"))
        return strengths == ["..."] and weaknesses == ["..."]

    def _parse_score(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        parsed = None
        if isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                parsed = float(match.group(0))
        if parsed is None:
            return None
        if 1.0 < parsed <= 100.0:
            parsed = parsed / 100.0
        return max(0.0, min(1.0, parsed))

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "pass", "passed"}:
                return True
            if lowered in {"false", "no", "0", "fail", "failed"}:
                return False
        return None

    def _parse_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def _fallback_evaluation(self, candidate: str, error: str) -> RSLEvaluation:
        normalized = _normalize_text(candidate)
        refusal_markers = ("i cannot", "i can't", "无法回答", "不能回答", "抱歉，我不能")
        if any(marker in normalized for marker in refusal_markers):
            score = 0.1
        elif len(candidate) < 80:
            score = 0.35
        elif len(candidate) < 240:
            score = 0.55
        else:
            score = 0.65
        return RSLEvaluation(
            score=score,
            passed=False,
            feedback="Evaluator output was not valid JSON; deterministic fallback scoring was used.",
            strengths=["The candidate is non-empty."] if candidate.strip() else [],
            weaknesses=[f"Structured evaluation failed: {error[:160]}"],
            dimension_scores={"clarity": score},
            evaluator_score=score,
            process_score=score,
            evaluator_backends=["local_fallback"],
        )

    def _sanitize_candidate(self, answer: str, request: RSLRunRequest) -> tuple[str, bool]:
        text = (answer or "").strip()
        changed = False
        needs_raw_json = any(check.kind == "json_valid" for check in request.external_checks)
        needs_python = any(check.kind == "python_syntax" for check in request.external_checks)
        forbids_fence = any(
            constraint.kind == "not_contains" and constraint.value == "```"
            for constraint in request.constraints
        ) or any(
            check.kind == "not_contains" and check.value == "```"
            for check in request.external_checks
        )
        if forbids_fence or needs_raw_json or needs_python:
            fenced = re.search(r"```(?:[a-zA-Z0-9_+.-]+)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if fenced:
                text = fenced.group(1).strip()
                changed = True
            elif "```" in text:
                text = "\n".join(line for line in text.splitlines() if "```" not in line).strip()
                changed = True
        if needs_raw_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                for index, character in enumerate(text):
                    if character not in "[{":
                        continue
                    try:
                        _, end = decoder.raw_decode(text[index:])
                    except json.JSONDecodeError:
                        continue
                    text = text[index:index + end].strip()
                    changed = True
                    break
        if needs_python:
            text = re.sub(r"^\s*```[a-zA-Z0-9_+.-]*\s*|\s*```\s*$", "", text).strip()
        return text, changed

    def _format_output_contract(self, request: RSLRunRequest) -> str:
        lines: list[str] = []
        for constraint in request.constraints:
            if constraint.kind == "not_contains":
                lines.append(f"- Do not include the exact text {constraint.value!r}.")
            elif constraint.kind == "contains":
                lines.append(f"- Include the exact text {constraint.value!r}.")
            elif constraint.kind == "line_count":
                lines.append(f"- Output exactly {constraint.value} lines.")
            elif constraint.kind == "min_lines":
                lines.append(f"- Output at least {constraint.value} lines.")
            elif constraint.kind == "max_lines":
                lines.append(f"- Output no more than {constraint.value} lines.")
            elif constraint.kind == "line_starts_with":
                lines.append(f"- Line {constraint.line} must start with {constraint.value!r}.")
            elif constraint.kind == "line_contains":
                lines.append(f"- Line {constraint.line} must contain {constraint.value!r}.")
            elif constraint.kind == "line_exact_chars":
                lines.append(f"- Line {constraint.line} must contain exactly {constraint.value} Han characters.")
        for check in request.external_checks:
            if check.kind == "json_valid":
                lines.append("- The entire answer must be valid raw JSON with no Markdown fence.")
            elif check.kind == "python_syntax":
                lines.append("- The answer must parse as valid Python source code.")
            elif check.kind == "contains_all":
                lines.append("- Include all of these values: " + ", ".join(repr(item) for item in check.values) + ".")
            elif check.kind == "not_contains":
                lines.append(f"- Do not include the exact text {check.value!r}.")
            elif check.kind == "contains":
                lines.append(f"- Include the exact text {check.value!r}.")
            elif check.kind == "regex":
                lines.append("- The whole answer must match an automated regular-expression check.")
            elif check.kind == "not_regex":
                lines.append("- The answer must avoid a forbidden automated regular-expression pattern.")
            elif check.kind == "min_chars":
                lines.append(f"- The answer must contain at least {check.value} characters.")
            elif check.kind == "max_chars":
                lines.append(f"- The answer must contain no more than {check.value} characters.")
        return "\n".join(lines)

    def _evaluate_constraints(
        self,
        candidate: str,
        constraints: list[RSLConstraint],
    ) -> tuple[float | None, list[str]]:
        if not constraints:
            return None, []
        evidence = []
        passed_count = 0
        for constraint in constraints:
            valid, detail = self._check_constraint(candidate, constraint)
            if valid:
                passed_count += 1
                evidence.append(f"PASS {constraint.name}")
            else:
                evidence.append(f"FAIL {constraint.name}: {detail}")
        return passed_count / len(constraints), evidence

    def _check_constraint(self, candidate: str, constraint: RSLConstraint) -> tuple[bool, str]:
        lines = candidate.splitlines()
        kind = constraint.kind
        value = constraint.value
        line_number = constraint.line
        if kind == "contains":
            return value in candidate, f"expected to contain {value!r}"
        if kind == "not_contains":
            return value not in candidate, f"expected not to contain {value!r}"
        if kind in {"line_count", "min_lines", "max_lines"}:
            try:
                expected = int(value)
            except ValueError:
                return False, f"invalid integer value {value!r}"
            if kind == "line_count":
                return len(lines) == expected, f"expected exactly {expected} lines, got {len(lines)}"
            if kind == "min_lines":
                return len(lines) >= expected, f"expected at least {expected} lines, got {len(lines)}"
            return len(lines) <= expected, f"expected at most {expected} lines, got {len(lines)}"
        if not line_number:
            return False, "line number is required"
        if line_number > len(lines):
            return False, f"line {line_number} does not exist"
        line = lines[line_number - 1].strip()
        if kind == "line_exact_chars":
            try:
                expected = int(value)
            except ValueError:
                return False, f"invalid integer value {value!r}"
            cjk_count = len(re.findall(r"[\u4e00-\u9fff]", line))
            actual = cjk_count if cjk_count else len(re.sub(r"\s+", "", line))
            return actual == expected, f"line {line_number} expected {expected} chars, got {actual}"
        if kind == "line_starts_with":
            return line.startswith(value), f"line {line_number} must start with {value!r}, got {line!r}"
        if kind == "line_contains":
            return value in line, f"line {line_number} must contain {value!r}, got {line!r}"
        return False, f"unsupported constraint kind {kind!r}"

    def _evaluate_external_checks(
        self,
        candidate: str,
        checks: list[RSLExternalCheck],
    ) -> tuple[float | None, list[str]]:
        if not checks:
            return None, []
        evidence = []
        passed_count = 0
        for check in checks:
            valid, detail = self._check_external(candidate, check)
            if valid:
                passed_count += 1
                evidence.append(f"PASS {check.name}")
            else:
                evidence.append(f"FAIL {check.name}: {detail}")
        return passed_count / len(checks), evidence

    def _check_external(self, candidate: str, check: RSLExternalCheck) -> tuple[bool, str]:
        kind = check.kind
        value = check.value or ""
        if kind == "contains":
            return value in candidate, f"expected to contain {value!r}"
        if kind == "not_contains":
            return value not in candidate, f"expected not to contain {value!r}"
        if kind == "regex":
            return re.search(value, candidate, re.DOTALL) is not None, f"regex did not match: {value!r}"
        if kind == "not_regex":
            return re.search(value, candidate, re.DOTALL) is None, f"forbidden regex matched: {value!r}"
        if kind == "json_valid":
            try:
                json.loads(candidate)
                return True, "valid JSON"
            except json.JSONDecodeError as exc:
                return False, f"invalid JSON: {exc}"
        if kind == "python_syntax":
            try:
                ast.parse(candidate)
                return True, "valid Python syntax"
            except SyntaxError as exc:
                return False, f"Python syntax error: {exc}"
        if kind == "min_chars":
            try:
                expected = int(value)
            except ValueError:
                return False, f"invalid integer value {value!r}"
            return len(candidate) >= expected, f"expected at least {expected} characters, got {len(candidate)}"
        if kind == "max_chars":
            try:
                expected = int(value)
            except ValueError:
                return False, f"invalid integer value {value!r}"
            return len(candidate) <= expected, f"expected at most {expected} characters, got {len(candidate)}"
        if kind == "contains_all":
            missing = [item for item in check.values if item not in candidate]
            return not missing, f"missing required values: {missing}"
        return False, f"unsupported external check kind {kind!r}"

    def _safety_flags(self, candidate: str) -> list[str]:
        return [name for name, pattern in SAFETY_PATTERNS.items() if re.search(pattern, candidate, re.DOTALL)]

    def _verification_mode(self, request: RSLRunRequest) -> str:
        if request.external_checks or request.holdout_checks:
            return "external_rules"
        if request.constraints:
            return "format_only"
        return "internal_only"

    def _make_round(
        self,
        *,
        round_index: int,
        strategy: str,
        candidates: list[RSLCandidate],
        memory_hits: list[str],
    ) -> RSLRound:
        scores = [candidate.evaluation.score for candidate in candidates]
        best = max(candidates, key=lambda item: item.evaluation.score)
        diversity, duplicate_rate = self._diversity_metrics([candidate.answer for candidate in candidates])
        for candidate in candidates:
            candidate.diversity_from_best = 1.0 - _jaccard(
                _similarity_tokens(candidate.answer),
                _similarity_tokens(best.answer),
            )
        return RSLRound(
            round_index=round_index,
            strategy=strategy,
            answer=best.answer,
            evaluation=best.evaluation,
            usage={
                "total_tokens": sum(
                    int((candidate.usage or {}).get("total_tokens") or 0)
                    for candidate in candidates
                )
            },
            latency_ms=sum(candidate.latency_ms for candidate in candidates),
            accepted=any(candidate.accepted for candidate in candidates),
            memory_hits=memory_hits,
            candidates=candidates,
            average_score=statistics.fmean(scores),
            score_std=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            diversity=diversity,
            duplicate_rate=duplicate_rate,
            best_candidate_index=best.candidate_index,
        )

    def _diversity_metrics(self, answers: list[str]) -> tuple[float, float]:
        if len(answers) < 2:
            return 1.0, 0.0
        token_sets = [_similarity_tokens(answer) for answer in answers]
        similarities = []
        duplicates = 0
        compared = 0
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                similarity = _jaccard(token_sets[i], token_sets[j])
                similarities.append(similarity)
                compared += 1
                if similarity >= 0.95:
                    duplicates += 1
        diversity = 1.0 - statistics.fmean(similarities) if similarities else 1.0
        duplicate_rate = duplicates / compared if compared else 0.0
        return max(0.0, min(1.0, diversity)), max(0.0, min(1.0, duplicate_rate))

    def _build_metrics(
        self,
        *,
        request: RSLRunRequest,
        baseline_score: float,
        best_score: float,
        all_candidates: list[RSLCandidate],
        rounds: list[RSLRound],
        holdout_score: float | None,
        budget: _BudgetManager,
        verified_pass: bool,
    ) -> RSLRunMetrics:
        scores = [candidate.evaluation.score for candidate in all_candidates]
        evaluator_scores = [candidate.evaluation.evaluator_score for candidate in all_candidates]
        external_scores = [
            candidate.evaluation.external_score
            for candidate in all_candidates
            if candidate.evaluation.external_score is not None
        ]
        improvement = max(0.0, best_score - baseline_score)
        verified_improvement = improvement if verified_pass else None
        mean_diversity = statistics.fmean([round_result.diversity for round_result in rounds]) if rounds else 0.0
        mean_duplicate = statistics.fmean([round_result.duplicate_rate for round_result in rounds]) if rounds else 0.0
        return RSLRunMetrics(
            baseline_score=baseline_score,
            best_score=best_score,
            improvement=improvement,
            verified_improvement=verified_improvement,
            sample_efficiency=improvement / budget.generation_calls if budget.generation_calls else 0.0,
            mean_candidate_score=statistics.fmean(scores) if scores else 0.0,
            score_std=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            mean_diversity=mean_diversity,
            mean_duplicate_rate=mean_duplicate,
            holdout_score=holdout_score,
            calibration_correlation=_pearson(evaluator_scores, external_scores),
            model_calls=budget.model_calls,
            generation_calls=budget.generation_calls,
            evaluator_calls=budget.evaluator_calls,
            total_tokens=budget.total_tokens,
        )

    def _build_learned_strategy(
        self,
        *,
        best_evaluation: RSLEvaluation,
        best_round: int,
        improvement: float,
    ) -> str:
        reusable_weaknesses = [
            item for item in best_evaluation.weaknesses
            if not item.startswith("Structured evaluation failed:")
        ]
        weaknesses = "; ".join(reusable_weaknesses[:3]).strip()
        if weaknesses:
            return (
                f"Best verified candidate was reached in round {best_round}. "
                f"Reuse this procedure for similar tasks, explicitly checking: {weaknesses}"
            )
        return (
            f"Best verified candidate was reached in round {best_round}. "
            f"Keep candidate search with independent checks (observed gain {improvement:.3f})."
        )

    async def _persist_round(self, run_id: str, round_result: RSLRound):
        await self.db._execute(
            """
            INSERT INTO rsl_rounds
            (run_id,round_index,strategy,answer,score,passed,feedback,strengths_json,
             weaknesses_json,dimensions_json,usage_json,latency_ms,accepted,memory_hits_json,
             candidates_json,average_score,score_std,diversity,duplicate_rate,best_candidate_index,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                round_result.round_index,
                round_result.strategy,
                round_result.answer,
                round_result.evaluation.score,
                1 if round_result.evaluation.passed else 0,
                round_result.evaluation.feedback,
                _json_dumps(round_result.evaluation.strengths),
                _json_dumps(round_result.evaluation.weaknesses),
                _json_dumps(round_result.evaluation.dimension_scores),
                _json_dumps(round_result.usage or {}),
                round_result.latency_ms,
                1 if round_result.accepted else 0,
                _json_dumps(round_result.memory_hits),
                _json_dumps([candidate.model_dump() for candidate in round_result.candidates]),
                round_result.average_score,
                round_result.score_std,
                round_result.diversity,
                round_result.duplicate_rate,
                round_result.best_candidate_index,
                _utc_now(),
            ),
        )
        for candidate in round_result.candidates:
            evaluation = candidate.evaluation
            await self.db._execute(
                """
                INSERT INTO rsl_candidates
                (run_id,round_index,candidate_index,answer,raw_answer,postprocessed,
                 content_hash,score,passed,evaluator_score,format_score,external_score,
                 process_score,feedback,safety_flags_json,evidence_json,dimensions_json,
                 usage_json,latency_ms,accepted,is_best,seed,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    round_result.round_index,
                    candidate.candidate_index,
                    candidate.answer,
                    candidate.raw_answer,
                    1 if candidate.postprocessed else 0,
                    candidate.content_hash,
                    evaluation.score,
                    1 if evaluation.passed else 0,
                    evaluation.evaluator_score,
                    evaluation.format_score,
                    evaluation.external_score,
                    evaluation.process_score,
                    evaluation.feedback,
                    _json_dumps(evaluation.safety_flags),
                    _json_dumps(evaluation.evidence),
                    _json_dumps(evaluation.dimension_scores),
                    _json_dumps(candidate.usage or {}),
                    candidate.latency_ms,
                    1 if candidate.accepted else 0,
                    1 if candidate.is_best else 0,
                    candidate.seed,
                    _utc_now(),
                ),
            )
        await self.db._conn.commit()

    async def _finish_run(
        self,
        *,
        run_id: str,
        status: str,
        completed_at: str,
        current_round: int,
        baseline_score: float,
        best_score: float,
        improvement: float,
        verified_improvement: float | None,
        best_round: int,
        learned_strategy: str,
        memory_used: int,
        memory_written: bool,
        memory_id: int | None,
        stopped_reason: str,
        verification_mode: str,
        metrics: RSLRunMetrics,
        controls: dict[str, float | None],
        budget_used: dict[str, int | float],
        duration_ms: int,
    ):
        await self.db._execute(
            """
            UPDATE rsl_runs
            SET completed_at=?,status=?,current_round=?,initial_score=?,baseline_score=?,best_score=?,
                improvement=?,verified_improvement=?,best_round=?,learned_strategy=?,memory_used=?,
                memory_written=?,memory_id=?,stopped_reason=?,verification_mode=?,metrics_json=?,
                controls_json=?,budget_json=?,duration_ms=?
            WHERE run_id=?
            """,
            (
                completed_at,
                status,
                current_round,
                baseline_score,
                baseline_score,
                best_score,
                improvement,
                verified_improvement,
                best_round,
                learned_strategy,
                memory_used,
                1 if memory_written else 0,
                memory_id,
                stopped_reason,
                verification_mode,
                _json_dumps(metrics.model_dump()),
                _json_dumps(controls),
                _json_dumps(budget_used),
                duration_ms,
                run_id,
            ),
        )
        await self.db._conn.commit()

    async def _store_verified_memory(
        self,
        *,
        request: RSLRunRequest,
        task_signature: str,
        lesson: str,
        score_before: float,
        score_after: float,
        source_run_id: str,
        best_candidate: RSLCandidate,
        holdout_score: float | None,
    ) -> tuple[int, bool] | None:
        evaluation = best_candidate.evaluation
        eligible = (
            evaluation.external_score is not None
            and evaluation.external_score == 1.0
            and (evaluation.format_score is None or evaluation.format_score == 1.0)
            and evaluation.process_score >= request.memory_min_score
            and evaluation.score >= request.memory_min_score
            and not evaluation.safety_flags
            and (holdout_score is None or holdout_score == 1.0)
        )
        if not eligible:
            return None
        previous = await self.db._fetchone(
            """
            SELECT * FROM rsl_memories
            WHERE model=? AND task_signature=? AND status IN ('candidate','active')
            ORDER BY version DESC LIMIT 1
            """,
            (request.model, task_signature),
        )
        previous_evidence = int(previous["evidence_count"] or 1) if previous else 0
        evidence_count = previous_evidence + 1 if previous else 1
        version = int(previous["version"] or 1) + 1 if previous else 1
        status = "active" if evidence_count >= request.memory_min_evidence else "candidate"
        previous_version_id = int(previous["id"]) if previous else None
        if status == "active" and previous and previous["status"] == "active":
            await self.db._execute("UPDATE rsl_memories SET status='superseded' WHERE id=?", (previous["id"],))
        cursor = await self.db._execute(
            """
            INSERT INTO rsl_memories
            (created_at,model,task_signature,task_example,lesson,score_gain,score_before,
             score_after,source_run_id,version,status,evidence_count,previous_version_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                _utc_now(),
                request.model,
                task_signature,
                request.task[:500],
                lesson,
                max(0.0, score_after - score_before),
                score_before,
                score_after,
                source_run_id,
                version,
                status,
                evidence_count,
                previous_version_id,
            ),
        )
        await self.db._conn.commit()
        return int(cursor.lastrowid), True

    async def _retrieve_memories(
        self,
        *,
        task: str,
        model: str,
        limit: int,
    ) -> list[RSLMemoryItem]:
        rows = await self.db._fetchall(
            """
            SELECT * FROM rsl_memories
            WHERE model=? AND status='active'
            ORDER BY version DESC
            LIMIT 100
            """,
            (model,),
        )
        if not rows:
            return []
        query_tokens = _similarity_tokens(task)
        ranked = []
        for row in rows:
            memory_tokens = _similarity_tokens(row["task_signature"])
            similarity = _jaccard(query_tokens, memory_tokens)
            if similarity <= 0:
                continue
            ranked.append((similarity, float(row["score_gain"] or 0.0), self._row_to_memory(row)))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:limit]]

    async def rollback_memory(self, memory_id: int, reason: str = "manual_rollback") -> RSLMemoryItem | None:
        row = await self.db._fetchone("SELECT * FROM rsl_memories WHERE id=?", (memory_id,))
        if not row:
            return None
        await self.db._execute(
            "UPDATE rsl_memories SET status='rolled_back',rollback_reason=?,rolled_back_at=? WHERE id=?",
            (reason, _utc_now(), memory_id),
        )
        previous_id = row["previous_version_id"]
        if previous_id:
            await self.db._execute(
                "UPDATE rsl_memories SET status='active' WHERE id=? AND status IN ('superseded','candidate')",
                (previous_id,),
            )
        await self.db._conn.commit()
        refreshed = await self.db._fetchone("SELECT * FROM rsl_memories WHERE id=?", (memory_id,))
        return self._row_to_memory(refreshed) if refreshed else None

    async def list_runs(self, limit: int = 20) -> list[RSLRunSummary]:
        rows = await self.db._fetchall(
            """
            SELECT run_id,created_at,completed_at,model,task,status,best_score,improvement,
                   verified_improvement,best_round,stopped_reason,duration_ms
            FROM rsl_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            RSLRunSummary(
                run_id=row["run_id"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                model=row["model"],
                task=row["task"],
                status=row["status"],
                best_score=row["best_score"],
                improvement=row["improvement"],
                verified_improvement=row["verified_improvement"],
                best_round=row["best_round"],
                stopped_reason=row["stopped_reason"],
                duration_ms=row["duration_ms"],
            )
            for row in rows
        ]

    async def get_run(self, run_id: str) -> RSLRunResponse | None:
        run = await self.db._fetchone("SELECT * FROM rsl_runs WHERE run_id=?", (run_id,))
        if not run:
            return None
        round_rows = await self.db._fetchall(
            "SELECT * FROM rsl_rounds WHERE run_id=? ORDER BY round_index ASC",
            (run_id,),
        )
        candidate_rows = await self.db._fetchall(
            "SELECT * FROM rsl_candidates WHERE run_id=? ORDER BY round_index ASC, candidate_index ASC",
            (run_id,),
        )
        candidates_by_round: dict[int, list[RSLCandidate]] = {}
        for row in candidate_rows:
            candidates_by_round.setdefault(int(row["round_index"]), []).append(self._row_to_candidate(row))
        rounds = [
            self._row_to_round(row, candidates_by_round.get(int(row["round_index"]), []))
            for row in round_rows
        ]
        best_answer = next(
            (item.answer for item in rounds if item.round_index == int(run["best_round"] or 0)),
            "",
        )
        if not best_answer and rounds:
            best_answer = rounds[-1].answer
        return RSLRunResponse(
            run_id=run["run_id"],
            created_at=run["created_at"],
            completed_at=run["completed_at"] or run["created_at"],
            task=run["task"],
            model=run["model"],
            evaluator_model=run["evaluator_model"],
            status=run["status"],
            baseline_score=float(run["baseline_score"] or run["initial_score"] or 0.0),
            initial_score=float(run["initial_score"] or 0.0),
            best_score=float(run["best_score"] or 0.0),
            improvement=float(run["improvement"] or 0.0),
            verified_improvement=run["verified_improvement"],
            verification_mode=run["verification_mode"] or "internal_only",
            best_round=int(run["best_round"] or 0),
            best_answer=best_answer,
            rounds=rounds,
            learned_strategy=run["learned_strategy"] or "",
            memory_used=int(run["memory_used"] or 0),
            memory_written=bool(run["memory_written"]),
            memory_id=run["memory_id"],
            stopped_reason=run["stopped_reason"] or "",
            verifier_version=run["verifier_version"] or "",
            prompt_version=run["prompt_version"] or "",
            controls=json.loads(run["controls_json"] or "{}"),
            metrics=self._parse_metrics(run),
            budget_used=json.loads(run["budget_json"] or "{}"),
            duration_ms=int(run["duration_ms"] or 0),
        )

    async def list_memories(
        self,
        *,
        model: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[RSLMemoryItem]:
        conditions = []
        params: list[Any] = []
        if model:
            conditions.append("model=?")
            params.append(model)
        if status:
            conditions.append("status=?")
            params.append(status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = await self.db._fetchall(
            f"SELECT * FROM rsl_memories {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        return [self._row_to_memory(row) for row in rows]

    def _parse_metrics(self, run) -> RSLRunMetrics:
        raw = json.loads(run["metrics_json"] or "{}")
        if raw:
            try:
                return RSLRunMetrics(**raw)
            except Exception:
                logger.warning("Failed to parse stored RSL metrics for %s", run["run_id"])
        return RSLRunMetrics(
            baseline_score=float(run["baseline_score"] or run["initial_score"] or 0.0),
            best_score=float(run["best_score"] or 0.0),
            improvement=float(run["improvement"] or 0.0),
            verified_improvement=run["verified_improvement"],
        )

    def _row_to_round(self, row, candidates: list[RSLCandidate]) -> RSLRound:
        if not candidates:
            candidates = [
                RSLCandidate(
                    candidate_index=0,
                    answer=row["answer"],
                    evaluation=RSLEvaluation(
                        score=float(row["score"] or 0.0),
                        passed=bool(row["passed"]),
                        feedback=row["feedback"] or "",
                        strengths=json.loads(row["strengths_json"] or "[]"),
                        weaknesses=json.loads(row["weaknesses_json"] or "[]"),
                        dimension_scores=json.loads(row["dimensions_json"] or "{}"),
                        evaluator_score=float(row["score"] or 0.0),
                        process_score=float(row["score"] or 0.0),
                    ),
                    usage=json.loads(row["usage_json"] or "{}") or None,
                    latency_ms=int(row["latency_ms"] or 0),
                    accepted=bool(row["accepted"]),
                    is_best=True,
                    content_hash=_content_hash(row["answer"]),
                )
            ]
        best = max(candidates, key=lambda item: item.evaluation.score)
        return RSLRound(
            round_index=int(row["round_index"]),
            strategy=row["strategy"],
            answer=best.answer,
            evaluation=best.evaluation,
            usage=json.loads(row["usage_json"] or "{}") or None,
            latency_ms=int(row["latency_ms"] or 0),
            accepted=bool(row["accepted"]),
            memory_hits=json.loads(row["memory_hits_json"] or "[]"),
            candidates=candidates,
            average_score=float(
                row["average_score"]
                or statistics.fmean([candidate.evaluation.score for candidate in candidates])
            ),
            score_std=float(row["score_std"] or 0.0),
            diversity=float(row["diversity"] or 0.0),
            duplicate_rate=float(row["duplicate_rate"] or 0.0),
            best_candidate_index=int(row["best_candidate_index"] or best.candidate_index),
        )

    def _row_to_candidate(self, row) -> RSLCandidate:
        return RSLCandidate(
            candidate_index=int(row["candidate_index"]),
            answer=row["answer"],
            evaluation=RSLEvaluation(
                score=float(row["score"] or 0.0),
                passed=bool(row["passed"]),
                feedback=row["feedback"] or "",
                strengths=[],
                weaknesses=[],
                dimension_scores=json.loads(row["dimensions_json"] or "{}"),
                evaluator_score=float(row["evaluator_score"] or 0.0),
                format_score=row["format_score"],
                external_score=row["external_score"],
                process_score=float(row["process_score"] or 0.0),
                evidence=json.loads(row["evidence_json"] or "[]"),
                safety_flags=json.loads(row["safety_flags_json"] or "[]"),
            ),
            usage=json.loads(row["usage_json"] or "{}") or None,
            latency_ms=int(row["latency_ms"] or 0),
            accepted=bool(row["accepted"]),
            is_best=bool(row["is_best"]),
            content_hash=row["content_hash"] or "",
            seed=row["seed"],
        )

    def _row_to_memory(self, row) -> RSLMemoryItem:
        return RSLMemoryItem(
            id=row["id"],
            created_at=row["created_at"],
            model=row["model"],
            lesson=row["lesson"],
            score_gain=float(row["score_gain"] or 0.0),
            score_before=row["score_before"],
            score_after=row["score_after"],
            source_run_id=row["source_run_id"],
            version=int(row["version"] or 1),
            status=row["status"] or "candidate",
            evidence_count=int(row["evidence_count"] or 1),
            previous_version_id=row["previous_version_id"],
            rollback_reason=row["rollback_reason"],
        )
