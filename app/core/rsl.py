from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.db import Database
from app.core.request_context import RequestContext
from app.core.router import RouterEngine
from app.schemas.chat import ChatMessage, ChatRequest
from app.schemas.rsl import (
    RSLEvaluation,
    RSLConstraint,
    RSLMemoryItem,
    RSLRound,
    RSLRunRequest,
    RSLRunResponse,
    RSLRunSummary,
)

logger = logging.getLogger("model_router.rsl")

GENERATOR_SYSTEM_PROMPT = """You are the execution module of a Recursive Self-Learning system.
Your job is to produce the best possible answer to the user's task.
Use lessons from previous runs and evaluator feedback when they are supplied.
Do not describe your reasoning or the learning process. Return only the final answer."""

EVALUATOR_SYSTEM_PROMPT = """You are a strict but fair evaluator for a Recursive Self-Learning system.
Analyze the candidate against the original task and criteria.
Return one JSON object only, with no markdown fences and no extra text.
Your response must contain your real judgment. Never copy field names or descriptions as values.
Required keys: score (number from 0 to 1), passed (boolean), feedback (one specific sentence), strengths (array of strings), weaknesses (array of strings), and dimension_scores (an object containing relevance, correctness, completeness, and clarity as numbers from 0 to 1).
A fully compliant answer should normally score 0.85 or higher. A partially compliant answer should score around 0.5 to 0.8. Use 0 only for an irrelevant or failed answer.
Set passed=true only when the candidate fully satisfies the task."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _similarity_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    latin = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]", normalized)
    return latin | {cjk[i] + cjk[i + 1] for i in range(max(0, len(cjk) - 1))}


class RSLEngine:
    """Recursive Self-Learning loop integrated with the existing ModelRouter.

    The loop deliberately updates an external strategy/memory layer rather than
    retraining model weights. Each round generates an answer, evaluates it,
    retains the best candidate, and feeds concrete weaknesses into the next
    generation round.
    """

    def __init__(self, db: Database, router: RouterEngine):
        self.db = db
        self.router = router

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
                best_score REAL DEFAULT 0,
                improvement REAL DEFAULT 0,
                best_round INTEGER DEFAULT 0,
                learned_strategy TEXT DEFAULT '',
                memory_used INTEGER DEFAULT 0,
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
                source_run_id TEXT NOT NULL
            )
            """
        )
        await self.db._execute(
            "CREATE INDEX IF NOT EXISTS idx_rsl_runs_created ON rsl_runs(created_at DESC)"
        )
        await self.db._execute(
            "CREATE INDEX IF NOT EXISTS idx_rsl_rounds_run ON rsl_rounds(run_id, round_index)"
        )
        await self.db._execute(
            "CREATE INDEX IF NOT EXISTS idx_rsl_memories_model ON rsl_memories(model, created_at DESC)"
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

        if request.persist:
            await self.db._execute(
                """
                INSERT INTO rsl_runs
                (run_id,created_at,subject_id,model,evaluator_model,task,task_signature,
                 status,current_round,target_score,memory_used)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
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
                ),
            )
            await self.db._conn.commit()

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
        best_answer = ""
        best_score = -1.0
        best_evaluation: RSLEvaluation | None = None
        best_round_index = 0
        initial_score = 0.0

        try:
            for round_index in range(request.max_rounds):
                strategy = "direct_generation" if round_index == 0 else "refine_from_feedback"
                candidate, generation_usage, generation_latency_ms = await self._generate(
                    task=request.task,
                    model=request.model,
                    criteria=request.criteria,
                    memories=memories,
                    previous_answer=best_answer,
                    previous_evaluation=best_evaluation,
                    round_index=round_index,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                evaluation = await self._evaluate(
                    task=request.task,
                    candidate=candidate,
                    criteria=request.criteria,
                    evaluator_model=evaluator_model,
                    constraints=request.constraints,
                )

                if round_index == 0:
                    initial_score = evaluation.score

                accepted = evaluation.score > best_score
                if accepted:
                    best_score = evaluation.score
                    best_answer = candidate
                    best_evaluation = evaluation
                    best_round_index = round_index + 1

                round_result = RSLRound(
                    round_index=round_index + 1,
                    strategy=strategy,
                    answer=candidate,
                    evaluation=evaluation,
                    usage=generation_usage,
                    latency_ms=generation_latency_ms,
                    accepted=accepted,
                    memory_hits=[item.lesson for item in memories],
                )
                rounds.append(round_result)

                if request.persist:
                    await self._insert_round(run_id, round_result)

                if evaluation.passed and evaluation.score >= request.target_score:
                    break

            if best_score < 0:
                raise RuntimeError("RSL finished without an evaluated candidate answer")

            improvement = max(0.0, best_score - initial_score)
            learned_strategy = self._build_learned_strategy(
                best_evaluation=best_evaluation,
                best_round=best_round_index,
                improvement=improvement,
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            completed_at = _utc_now()

            if request.persist and improvement >= 0.03 and best_evaluation is not None:
                await self._insert_memory(
                    model=request.model,
                    task=request.task,
                    task_signature=task_signature,
                    lesson=learned_strategy,
                    score_gain=improvement,
                    source_run_id=run_id,
                )

            if request.persist:
                await self.db._execute(
                    """
                    UPDATE rsl_runs
                    SET completed_at=?,status='completed',current_round=?,initial_score=?,
                        best_score=?,improvement=?,best_round=?,learned_strategy=?,
                        memory_used=?,duration_ms=?
                    WHERE run_id=?
                    """,
                    (
                        completed_at,
                        len(rounds),
                        initial_score,
                        best_score,
                        improvement,
                        best_round_index,
                        learned_strategy,
                        len(memories),
                        duration_ms,
                        run_id,
                    ),
                )
                await self.db._conn.commit()

            return RSLRunResponse(
                run_id=run_id,
                created_at=created_at,
                completed_at=completed_at,
                task=request.task,
                model=request.model,
                evaluator_model=evaluator_model,
                status="completed",
                initial_score=initial_score,
                best_score=best_score,
                improvement=improvement,
                best_round=best_round_index,
                best_answer=best_answer,
                rounds=rounds,
                learned_strategy=learned_strategy,
                memory_used=len(memories),
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            if request.persist:
                await self.db._execute(
                    """
                    UPDATE rsl_runs
                    SET completed_at=?,status='failed',current_round=?,duration_ms=?,error=?
                    WHERE run_id=?
                    """,
                    (_utc_now(), len(rounds), duration_ms, str(exc)[:1000], run_id),
                )
                await self.db._conn.commit()
            logger.exception("RSL run %s failed", run_id)
            raise

    async def _generate(
        self,
        *,
        task: str,
        model: str,
        criteria: str,
        memories: list[RSLMemoryItem],
        previous_answer: str,
        previous_evaluation: RSLEvaluation | None,
        round_index: int,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict | None, int]:
        sections = [
            f"ORIGINAL TASK:\n{task}",
            f"SUCCESS CRITERIA:\n{criteria}",
        ]
        if memories:
            lesson_text = "\n".join(f"- {item.lesson}" for item in memories)
            sections.append(f"REUSABLE LESSONS FROM PREVIOUS RUNS:\n{lesson_text}")
        if round_index == 0:
            sections.append("Produce a complete first answer. Prefer concrete details over vague claims.")
        else:
            sections.append(f"PREVIOUS BEST ANSWER:\n{previous_answer}")
            if previous_evaluation:
                weaknesses = "\n".join(
                    f"- {item}" for item in previous_evaluation.weaknesses
                ) or "- No explicit weakness was returned."
                sections.append(
                    "EVALUATOR FEEDBACK:\n"
                    f"{previous_evaluation.feedback}\n"
                    f"WEAKNESSES TO FIX:\n{weaknesses}"
                )
            sections.append(
                "Rewrite the answer from scratch, fixing every listed weakness while preserving strengths."
            )

        messages = [
            ChatMessage(role="system", content=GENERATOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n\n".join(sections)),
        ]
        content, usage, latency_ms = await self._call_model(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content, usage, latency_ms

    async def _evaluate(
        self,
        *,
        task: str,
        candidate: str,
        criteria: str,
        evaluator_model: str,
        constraints: list[RSLConstraint] | None = None,
    ) -> RSLEvaluation:
        evaluation_prompt = (
            f"ORIGINAL TASK:\n{task}\n\n"
            f"SUCCESS CRITERIA:\n{criteria}\n\n"
            f"CANDIDATE ANSWER:\n{candidate}\n\n"
            "Evaluate only from the information above."
        )
        messages = [
            ChatMessage(role="system", content=EVALUATOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content=evaluation_prompt),
        ]
        content, _, _ = await self._call_model(
            model=evaluator_model,
            messages=messages,
            temperature=0.0,
            max_tokens=400,
        )
        try:
            evaluation = self._parse_evaluation(content)
        except Exception as exc:
            logger.warning("RSL evaluator returned invalid structured output: %s", exc)
            evaluation = self._fallback_evaluation(candidate, str(exc))
        return self._apply_constraints(evaluation, candidate, constraints or [])

    def _apply_constraints(
        self,
        evaluation: RSLEvaluation,
        candidate: str,
        constraints: list[RSLConstraint],
    ) -> RSLEvaluation:
        if not constraints:
            return evaluation

        violations: list[str] = []
        passed_count = 0
        for constraint in constraints:
            valid, detail = self._check_constraint(candidate, constraint)
            if valid:
                passed_count += 1
            else:
                violations.append(f"{constraint.name}: {detail}")

        constraint_score = passed_count / len(constraints)
        combined_score = round((evaluation.score + constraint_score) / 2.0, 4)
        feedback_parts = [evaluation.feedback]
        if violations:
            feedback_parts.append("Hard constraint violations: " + "; ".join(violations))
        else:
            feedback_parts.append("All hard constraints passed.")

        dimensions = dict(evaluation.dimension_scores)
        dimensions["constraints"] = constraint_score
        return RSLEvaluation(
            score=combined_score,
            passed=evaluation.passed and not violations and constraint_score == 1.0,
            feedback=" ".join(part for part in feedback_parts if part).strip(),
            strengths=list(evaluation.strengths),
            weaknesses=[*evaluation.weaknesses, *violations],
            dimension_scores=dimensions,
        )

    def _check_constraint(
        self,
        candidate: str,
        constraint: RSLConstraint,
    ) -> tuple[bool, str]:
        lines = candidate.splitlines()
        kind = constraint.kind
        value = constraint.value
        line_number = constraint.line

        if kind == "contains":
            ok = value in candidate
            return ok, f"expected to contain {value!r}"
        if kind == "not_contains":
            ok = value not in candidate
            return ok, f"expected not to contain {value!r}"
        if kind in {"line_count", "min_lines", "max_lines"}:
            try:
                expected = int(value)
            except ValueError:
                return False, f"invalid integer value {value!r}"
            if kind == "line_count":
                ok = len(lines) == expected
                return ok, f"expected exactly {expected} lines, got {len(lines)}"
            if kind == "min_lines":
                ok = len(lines) >= expected
                return ok, f"expected at least {expected} lines, got {len(lines)}"
            ok = len(lines) <= expected
            return ok, f"expected at most {expected} lines, got {len(lines)}"
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
            ok = line.startswith(value)
            return ok, f"line {line_number} must start with {value!r}, got {line!r}"
        if kind == "line_contains":
            ok = value in line
            return ok, f"line {line_number} must contain {value!r}, got {line!r}"
        return False, f"unsupported constraint kind {kind!r}"

    async def _call_model(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict | None, int]:
        provider, model_config = await self.router.route(model)
        request = ChatRequest(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
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

        passed_raw = payload.get("passed")
        passed = self._parse_bool(passed_raw)
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
        )

    def _looks_like_placeholder(self, payload: dict) -> bool:
        feedback = str(payload.get("feedback") or "").strip().lower()
        if feedback in {"short concrete feedback", "specific sentence", "..."}:
            return True
        strengths = self._parse_string_list(payload.get("strengths"))
        weaknesses = self._parse_string_list(payload.get("weaknesses"))
        if strengths == ["..."] and weaknesses == ["..."]:
            return True
        return False
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
        refusal_markers = (
            "i cannot",
            "i can't",
            "无法回答",
            "不能回答",
            "抱歉，我不能",
        )
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
        )

    def _build_learned_strategy(
        self,
        *,
        best_evaluation: RSLEvaluation | None,
        best_round: int,
        improvement: float,
    ) -> str:
        if best_evaluation is None:
            return "No reusable strategy was recorded."
        weaknesses = "; ".join(best_evaluation.weaknesses[:3]).strip()
        if weaknesses:
            return (
                f"Best result was reached in round {best_round}; for similar tasks, "
                f"explicitly address these weaknesses: {weaknesses}"
            )
        return (
            f"Best result was reached in round {best_round}; keep the evaluation-driven "
            f"rewrite strategy (observed score gain {improvement:.2f})."
        )

    async def _insert_round(self, run_id: str, round_result: RSLRound):
        evaluation = round_result.evaluation
        await self.db._execute(
            """
            INSERT INTO rsl_rounds
            (run_id,round_index,strategy,answer,score,passed,feedback,strengths_json,
             weaknesses_json,dimensions_json,usage_json,latency_ms,accepted,memory_hits_json,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                round_result.round_index,
                round_result.strategy,
                round_result.answer,
                evaluation.score,
                1 if evaluation.passed else 0,
                evaluation.feedback,
                _json_dumps(evaluation.strengths),
                _json_dumps(evaluation.weaknesses),
                _json_dumps(evaluation.dimension_scores),
                _json_dumps(round_result.usage or {}),
                round_result.latency_ms,
                1 if round_result.accepted else 0,
                _json_dumps(round_result.memory_hits),
                _utc_now(),
            ),
        )
        await self.db._conn.commit()

    async def _insert_memory(
        self,
        *,
        model: str,
        task: str,
        task_signature: str,
        lesson: str,
        score_gain: float,
        source_run_id: str,
    ):
        await self.db._execute(
            """
            INSERT INTO rsl_memories
            (created_at,model,task_signature,task_example,lesson,score_gain,source_run_id)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                _utc_now(),
                model,
                task_signature,
                task[:500],
                lesson,
                score_gain,
                source_run_id,
            ),
        )
        await self.db._conn.commit()

    async def _retrieve_memories(
        self,
        *,
        task: str,
        model: str,
        limit: int,
    ) -> list[RSLMemoryItem]:
        rows = await self.db._fetchall(
            """
            SELECT id,created_at,model,task_signature,lesson,score_gain,source_run_id
            FROM rsl_memories
            WHERE model=?
            ORDER BY created_at DESC
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
            union = query_tokens | memory_tokens
            similarity = len(query_tokens & memory_tokens) / len(union) if union else 0.0
            if similarity <= 0:
                continue
            ranked.append(
                (
                    similarity,
                    float(row["score_gain"] or 0.0),
                    RSLMemoryItem(
                        id=row["id"],
                        created_at=row["created_at"],
                        model=row["model"],
                        lesson=row["lesson"],
                        score_gain=float(row["score_gain"] or 0.0),
                        source_run_id=row["source_run_id"],
                    ),
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:limit]]

    async def list_runs(self, limit: int = 20) -> list[RSLRunSummary]:
        rows = await self.db._fetchall(
            """
            SELECT run_id,created_at,completed_at,model,task,status,best_score,
                   improvement,best_round,duration_ms
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
                best_round=row["best_round"],
                duration_ms=row["duration_ms"],
            )
            for row in rows
        ]

    async def get_run(self, run_id: str) -> RSLRunResponse | None:
        run = await self.db._fetchone("SELECT * FROM rsl_runs WHERE run_id=?", (run_id,))
        if not run:
            return None
        rounds = await self.db._fetchall(
            "SELECT * FROM rsl_rounds WHERE run_id=? ORDER BY round_index ASC",
            (run_id,),
        )
        parsed_rounds = [self._row_to_round(row) for row in rounds]
        best_answer = next(
            (item.answer for item in parsed_rounds if item.round_index == run["best_round"]),
            "",
        )
        if not best_answer and parsed_rounds:
            best_answer = parsed_rounds[-1].answer
        return RSLRunResponse(
            run_id=run["run_id"],
            created_at=run["created_at"],
            completed_at=run["completed_at"] or run["created_at"],
            task=run["task"],
            model=run["model"],
            evaluator_model=run["evaluator_model"],
            status=run["status"],
            initial_score=float(run["initial_score"] or 0.0),
            best_score=float(run["best_score"] or 0.0),
            improvement=float(run["improvement"] or 0.0),
            best_round=int(run["best_round"] or 0),
            best_answer=best_answer,
            rounds=parsed_rounds,
            learned_strategy=run["learned_strategy"] or "",
            memory_used=int(run["memory_used"] or 0),
            duration_ms=int(run["duration_ms"] or 0),
        )

    async def list_memories(
        self,
        *,
        model: str | None = None,
        limit: int = 20,
    ) -> list[RSLMemoryItem]:
        if model:
            rows = await self.db._fetchall(
                """
                SELECT id,created_at,model,lesson,score_gain,source_run_id
                FROM rsl_memories WHERE model=? ORDER BY created_at DESC LIMIT ?
                """,
                (model, limit),
            )
        else:
            rows = await self.db._fetchall(
                """
                SELECT id,created_at,model,lesson,score_gain,source_run_id
                FROM rsl_memories ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            )
        return [
            RSLMemoryItem(
                id=row["id"],
                created_at=row["created_at"],
                model=row["model"],
                lesson=row["lesson"],
                score_gain=float(row["score_gain"] or 0.0),
                source_run_id=row["source_run_id"],
            )
            for row in rows
        ]

    def _row_to_round(self, row) -> RSLRound:
        return RSLRound(
            round_index=row["round_index"],
            strategy=row["strategy"],
            answer=row["answer"],
            evaluation=RSLEvaluation(
                score=float(row["score"] or 0.0),
                passed=bool(row["passed"]),
                feedback=row["feedback"] or "",
                strengths=json.loads(row["strengths_json"] or "[]"),
                weaknesses=json.loads(row["weaknesses_json"] or "[]"),
                dimension_scores=json.loads(row["dimensions_json"] or "{}"),
            ),
            usage=json.loads(row["usage_json"] or "{}") or None,
            latency_ms=int(row["latency_ms"] or 0),
            accepted=bool(row["accepted"]),
            memory_hits=json.loads(row["memory_hits_json"] or "[]"),
        )
