from typing import Literal

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


class RSLRunRequest(BaseModel):
    task: str = Field(min_length=3, max_length=20_000)
    model: str = "qwen2.5-1.5b-instruct"
    evaluator_model: str | None = None
    max_rounds: int = Field(default=3, ge=1, le=5)
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)
    criteria: str = (
        "correctness, relevance, completeness, clarity, and direct fulfillment "
        "of the user's requested format"
    )
    use_memory: bool = True
    memory_limit: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=64, le=2048)
    constraints: list[RSLConstraint] = Field(default_factory=list)
    persist: bool = True


class RSLEvaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)


class RSLRound(BaseModel):
    round_index: int
    strategy: str
    answer: str
    evaluation: RSLEvaluation
    usage: dict | None = None
    latency_ms: int
    accepted: bool
    memory_hits: list[str] = Field(default_factory=list)


class RSLRunResponse(BaseModel):
    run_id: str
    created_at: str
    completed_at: str
    task: str
    model: str
    evaluator_model: str
    status: Literal["running", "completed", "failed"]
    initial_score: float
    best_score: float
    improvement: float
    best_round: int
    best_answer: str
    rounds: list[RSLRound]
    learned_strategy: str
    memory_used: int
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
    best_round: int | None = None
    duration_ms: int | None = None


class RSLRunList(BaseModel):
    runs: list[RSLRunSummary]


class RSLMemoryItem(BaseModel):
    id: int
    created_at: str
    model: str
    lesson: str
    score_gain: float
    source_run_id: str


class RSLMemoryList(BaseModel):
    memories: list[RSLMemoryItem]
