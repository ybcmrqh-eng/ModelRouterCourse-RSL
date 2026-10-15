import json
import time
import unittest

from app.core.config import ModelConfig
from app.core.db import Database
from app.core.rsl import EVALUATOR_SYSTEM_PROMPT, RSLEngine
from app.schemas.chat import ChatChoice, ChatMessage, ChatResponse, Usage
from app.schemas.rsl import RSLConstraint, RSLExternalCheck, RSLRunBudget, RSLRunRequest


class FakeProvider:
    async def chat_completions(self, request, model_config):
        system = request.messages[0].content
        user = request.messages[1].content
        if system == EVALUATOR_SYSTEM_PROMPT:
            good = "good answer" in user
            score = 0.95 if good else 0.45
            content = json.dumps(
                {
                    "score": score,
                    "passed": good,
                    "feedback": "The answer is complete." if good else "The answer is incomplete.",
                    "strengths": ["clear structure"] if good else [],
                    "weaknesses": [] if good else ["missing required content"],
                    "dimension_scores": {
                        "relevance": score,
                        "correctness": score,
                        "completeness": score,
                        "clarity": score,
                    },
                }
            )
        elif "PARENT CANDIDATE" in user:
            if "variant 1" in user:
                content = "good answer\nsecond line"
            else:
                content = "good answer\nthird line"
        else:
            content = "bad answer"
        return ChatResponse(
            id="fake-response",
            created=int(time.time()),
            model=model_config.name,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )


class FakeRouter:
    def __init__(self):
        self.provider = FakeProvider()
        self.config = ModelConfig(name="fake-model", routing_keys=["fake-model"])

    async def route(self, model):
        return self.provider, self.config


def base_request(**overrides):
    values = dict(
        task="Return a two-line answer containing good answer.",
        model="fake-model",
        evaluator_model="fake-model",
        max_rounds=2,
        candidates_per_round=2,
        top_k=2,
        patience=1,
        target_score=0.9,
        memory_min_evidence=2,
        memory_min_score=0.9,
        constraints=[RSLConstraint(name="two lines", kind="line_count", value="2")],
        external_checks=[RSLExternalCheck(name="contains good answer", kind="contains", value="good answer")],
    )
    values.update(overrides)
    return RSLRunRequest(**values)


class RSLEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = Database(":memory:")
        await self.db.connect()
        await self.db.migrate()
        self.engine = RSLEngine(self.db, FakeRouter())
        await self.engine.initialize()

    async def asyncTearDown(self):
        await self.db.close()

    async def test_candidate_search_records_controls_metrics_and_trace(self):
        result = await self.engine.run(base_request())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.verification_mode, "external_rules")
        self.assertGreater(result.best_score, result.baseline_score)
        self.assertIsNotNone(result.verified_improvement)
        self.assertEqual(result.rounds[0].strategy, "baseline_single_generation")
        self.assertEqual(len(result.rounds[0].candidates), 1)
        self.assertGreaterEqual(result.rounds[0].evaluation.external_score, 0)
        self.assertEqual(result.rounds[0].evaluation.external_score, 0.0)
        self.assertEqual(result.rounds[1].strategy, "candidate_search_topk")
        self.assertEqual(len(result.rounds[1].candidates), 2)
        self.assertIn("single_generation", result.controls)
        self.assertIn("generate_evaluate_revise", result.controls)
        self.assertIn("candidate_search_topk", result.controls)
        self.assertGreaterEqual(result.metrics.model_calls, 5)
        self.assertIsNotNone(result.metrics.mean_diversity)

        stored = await self.engine.get_run(result.run_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.best_answer, result.best_answer)
        self.assertEqual(len(stored.rounds[1].candidates), 2)

    async def test_memory_needs_two_verified_runs_and_can_roll_back(self):
        first = await self.engine.run(base_request(use_memory=False))
        self.assertTrue(first.memory_written)
        memories = await self.engine.list_memories(model="fake-model")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].status, "candidate")
        self.assertEqual(memories[0].evidence_count, 1)

        second = await self.engine.run(base_request(use_memory=False))
        self.assertTrue(second.memory_written)
        memories = await self.engine.list_memories(model="fake-model", status="active")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].status, "active")
        self.assertEqual(memories[0].evidence_count, 2)

        rolled_back = await self.engine.rollback_memory(memories[0].id, reason="test_rollback")
        self.assertIsNotNone(rolled_back)
        self.assertEqual(rolled_back.status, "rolled_back")
        active = await self.engine.list_memories(model="fake-model", status="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].id, memories[0].previous_version_id)

    async def test_budget_stops_before_search(self):
        request = base_request(
            candidates_per_round=2,
            max_rounds=2,
            budget=RSLRunBudget(max_model_calls=2, max_total_tokens=1000, max_duration_seconds=30),
        )
        result = await self.engine.run(request)
        self.assertEqual(result.stopped_reason, "budget_model_calls")
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.rounds[0].strategy, "baseline_single_generation")

    async def test_holdout_failure_blocks_memory_write(self):
        request = base_request(
            holdout_checks=[
                RSLExternalCheck(name="held out failure", kind="not_contains", value="good answer"),
            ]
        )
        result = await self.engine.run(request)
        self.assertFalse(result.memory_written)
        self.assertEqual(result.metrics.holdout_score, 0.0)
        self.assertIn("Holdout check failed", result.rounds[-1].evaluation.weaknesses[0])


if __name__ == "__main__":
    unittest.main()
