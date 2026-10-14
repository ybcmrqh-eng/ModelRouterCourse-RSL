import json
import time
import unittest

from app.core.db import Database
from app.core.rsl import EVALUATOR_SYSTEM_PROMPT, RSLEngine
from app.core.config import ModelConfig
from app.schemas.chat import ChatChoice, ChatMessage, ChatResponse, Usage
from app.schemas.rsl import RSLConstraint, RSLRunRequest


class FakeProvider:
    async def chat_completions(self, request, model_config):
        system = request.messages[0].content
        user = request.messages[1].content
        if system == EVALUATOR_SYSTEM_PROMPT:
            good = "good answer" in user
            score = 0.92 if good else 0.45
            content = json.dumps(
                {
                    "score": score,
                    "passed": good,
                    "feedback": "The answer improved." if good else "The answer is incomplete.",
                    "strengths": ["clear"] if good else [],
                    "weaknesses": [] if good else ["missing required structure"],
                    "dimension_scores": {
                        "relevance": score,
                        "correctness": score,
                        "completeness": score,
                        "clarity": score,
                    },
                }
            )
        else:
            content = "good answer\nsecond line" if "PREVIOUS BEST ANSWER" in user else "bad answer"
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


class RSLEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = Database(":memory:")
        await self.db.connect()
        await self.db.migrate()
        self.engine = RSLEngine(self.db, FakeRouter())
        await self.engine.initialize()

    async def asyncTearDown(self):
        await self.db.close()

    async def test_recursive_learning_improves_score_and_persists_memory(self):
        request = RSLRunRequest(
            task="Return a two-line answer.",
            model="fake-model",
            evaluator_model="fake-model",
            max_rounds=2,
            target_score=0.9,
            constraints=[
                RSLConstraint(name="two lines", kind="line_count", value="2"),
            ],
        )
        result = await self.engine.run(request)
        self.assertEqual(result.status, "completed")
        self.assertGreater(result.best_score, result.initial_score)
        self.assertEqual(result.best_round, 2)
        self.assertEqual(len(result.rounds), 2)
        self.assertTrue(result.rounds[0].evaluation.weaknesses)
        self.assertIn("Hard constraint violations", result.rounds[0].evaluation.feedback)
        self.assertTrue(result.rounds[1].evaluation.passed)

        stored = await self.engine.get_run(result.run_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.best_answer, result.best_answer)

        memories = await self.engine.list_memories(model="fake-model")
        self.assertEqual(len(memories), 1)


if __name__ == "__main__":
    unittest.main()
