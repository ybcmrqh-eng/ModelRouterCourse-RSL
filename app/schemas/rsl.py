from typing import Any, Literal

from pydantic import BaseModel, Field


class RSLConstraint(BaseModel):
    name: str
    kind: Literal[
        "contains",
        "not_contains",
        "line_count",
        "min_lines",
        "max_lines",
        "line_exact_chars",
        "line_starts_with",
        "line_contains",
    ]
    value: str
    line: int | None = Field(default=None, ge=1)


class RSLExternalCheck(BaseModel):
    """Deterministic check evaluated independently from the language-model judge."""

    name: str
    kind: Literal[
        "contains",
        "not_contains",
        "regex",
        "not_regex",
        "json_valid",
        "python_syntax",
        "min_chars",
        "max_chars",
        "contains_all",
    ]
    value: str | None = None
    values: list[str] = Field(default_factory=list)


class RSLRunBudget(BaseModel):
    max_model_calls: int = Field(default=32, ge=2, le=200)
    max_total_tokens: int = Field(default=40_000, ge=100, le=500_000)
    max_duration_seconds: int = Field(default=600, ge=5, le=3_600)


class RSLRunRequest(BaseModel):
    task: str = Field(min_length=3, max_length=20_000)
    model: str = "qwen2.5-1.5b-instruct"
    evaluator_model: str | None = None
    evaluator_backend: Literal["auto", "local", "typesafe"] = "auto"
    typesafe_model: str = "jev-latest"
    max_rounds: int = Field(default=3, ge=1, le=5)
    candidates_per_round: int = Field(default=3, ge=1, le=5)
    top_k: int = Field(default=2, ge=1, le=5)
    patience: int = Field(default=1, ge=1, le=5)
    baseline_enabled: bool = True
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)
    criteria: str = (
        "correctness, relevance, completeness, clarity, and direct fulfillment "
        "of the user's requested format"
    )
    use_memory: bool = True
    memory_limit: int = Field(default=3, ge=0, le=10)
    memory_min_evidence: int = Field(default=2, ge=1, le=10)
    memory_min_score: float = Field(default=0.9, ge=0.0, le=1.0)
    memory_write_enabled: bool = True
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    diversity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=64, le=2048)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    constraints: list[RSLConstraint] = Field(default_factory=list)
    external_checks: list[RSLExternalCheck] = Field(default_factory=list)
    holdout_checks: list[RSLExternalCheck] = Field(default_factory=list)
    enable_code_execution: bool = False
    budget: RSLRunBudget = Field(default_factory=RSLRunBudget)
    persist: bool = True


class RSLEvaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    evaluator_score: float = Field(default=0.0, ge=0.0, le=1.0)
    format_score: float | None = Field(default=None, ge=0.0, le=1.0)
    external_score: float | None = Field(default=None, ge=0.0, le=1.0)
    process_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    evaluator_backends: list[str] = Field(default_factory=list)


class RSLCandidate(BaseModel):
    candidate_index: int
    answer: str
    raw_answer: str | None = None
    postprocessed: bool = False
    evaluation: RSLEvaluation
    usage: dict | None = None
    latency_ms: int
    accepted: bool = False
    is_best: bool = False
    content_hash: str = ""
    diversity_from_best: float | None = None
    seed: int | None = None


class RSLRound(BaseModel):
    round_index: int
    strategy: str
    answer: str
    evaluation: RSLEvaluation
    usage: dict | None = None
    latency_ms: int
    accepted: bool
    memory_hits: list[str] = Field(default_factory=list)
    candidates: list[RSLCandidate] = Field(default_factory=list)
    average_score: float = 0.0
    score_std: float = 0.0
    diversity: float = 0.0
    duplicate_rate: float = 0.0
    best_candidate_index: int = 0


class RSLRunMetrics(BaseModel):
    baseline_score: float
    best_score: float
    improvement: float
    verified_improvement: float | None = None
    sample_efficiency: float = 0.0
    mean_candidate_score: float = 0.0
    score_std: float = 0.0
    mean_diversity: float = 0.0
    mean_duplicate_rate: float = 0.0
    holdout_score: float | None = None
    calibration_correlation: float | None = None
    model_calls: int = 0
    generation_calls: int = 0
    evaluator_calls: int = 0
    total_tokens: int = 0


class RSLRunResponse(BaseModel):
    run_id: str
    created_at: str
    completed_at: str
    task: str
    model: str
    evaluator_model: str
    status: Literal["running", "completed", "failed"]
    baseline_score: float = 0.0
    initial_score: float = 0.0
    best_score: float = 0.0
    improvement: float = 0.0
    verified_improvement: float | None = None
    verification_mode: str = "internal_only"
    best_round: int
    best_answer: str
    rounds: list[RSLRound]
    learned_strategy: str
    memory_used: int
    memory_written: bool = False
    memory_id: int | None = None
    stopped_reason: str = "max_rounds"
    verifier_version: str = ""
    prompt_version: str = ""
    controls: dict[str, float | None] = Field(default_factory=dict)
    metrics: RSLRunMetrics
    budget_used: dict[str, int | float] = Field(default_factory=dict)
    duration_ms: int


class RSLRunSummary(BaseModel):
    run_id: str
    created_at: str
    completed_at: str | None = None
    model: str
    task: str
    status: str
    best_score: float | None = None
    improvement: float | None = None
    verified_improvement: float | None = None
    best_round: int | None = None
    stopped_reason: str | None = None
    duration_ms: int | None = None


class RSLRunList(BaseModel):
    runs: list[RSLRunSummary]


class RSLMemoryItem(BaseModel):
    id: int
    created_at: str
    model: str
    lesson: str
    score_gain: float
    score_before: float | None = None
    score_after: float | None = None
    source_run_id: str
    version: int = 1
    status: str = "candidate"
    evidence_count: int = 1
    previous_version_id: int | None = None
    rollback_reason: str | None = None


class RSLMemoryList(BaseModel):
    memories: list[RSLMemoryItem]


class RSLMemoryRollbackRequest(BaseModel):
    reason: str = "manual_rollback"
