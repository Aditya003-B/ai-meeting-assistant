"""Offline evaluation pipeline for RAG and extraction accuracy."""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.extractor import extraction_agent
from app.agents.rag import rag_chain
from app.core.exceptions import AppException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Offline evaluation pipeline for RAG and extraction."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.golden_sets_dir = Path("golden_sets")
        self.results_dir = self.golden_sets_dir / "eval_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_rag(self, db: AsyncSession) -> float:
        """Evaluate RAG retrieval accuracy using golden set."""
        try:
            logger.info("Starting RAG evaluation")

            rag_eval_path = self.golden_sets_dir / "rag_eval.json"
            if not rag_eval_path.exists():
                logger.warning("RAG eval golden set not found")
                return 0.0

            with open(rag_eval_path) as f:
                golden_set = json.load(f)

            if not golden_set:
                return 0.0

            hits = 0
            total = len(golden_set)

            for item in golden_set:
                question = item["question"]
                expected_meeting_title = item.get("source_meeting_title")

                _, sources = await rag_chain.answer_question(
                    db=db,
                    question=question,
                    top_k=5,
                )

                if expected_meeting_title:
                    for source in sources:
                        if expected_meeting_title.lower() in source.title.lower():
                            hits += 1
                            break

            accuracy = hits / total if total > 0 else 0.0

            logger.info(f"RAG accuracy: {accuracy:.2%} ({hits}/{total})")

            return accuracy

        except Exception as e:
            logger.error(f"RAG evaluation failed: {str(e)}")
            raise AppException(
                message="Failed to evaluate RAG",
                details={"error": str(e)},
            ) from e

    async def evaluate_extraction(self) -> float:
        """Evaluate extraction F1 score using golden set."""
        try:
            logger.info("Starting extraction evaluation")

            extraction_eval_path = self.golden_sets_dir / "extraction_eval.json"
            if not extraction_eval_path.exists():
                logger.warning("Extraction eval golden set not found")
                return 0.0

            with open(extraction_eval_path) as f:
                golden_set = json.load(f)

            if not golden_set:
                return 0.0

            total_precision = 0.0
            total_recall = 0.0
            count = 0

            for item in golden_set:
                transcript = item["transcript"]
                expected_items = item["expected_action_items"]

                result = await extraction_agent.extract(transcript)
                extracted_items = result["action_items"]

                expected_tasks = {
                    item.get("task", "").lower() for item in expected_items
                }
                extracted_tasks = {
                    item.get("task", "").lower() for item in extracted_items
                }

                if extracted_tasks:
                    true_positives = len(expected_tasks & extracted_tasks)
                    precision = true_positives / len(extracted_tasks)
                    total_precision += precision

                if expected_tasks:
                    true_positives = len(expected_tasks & extracted_tasks)
                    recall = true_positives / len(expected_tasks)
                    total_recall += recall

                count += 1

            avg_precision = total_precision / count if count > 0 else 0.0
            avg_recall = total_recall / count if count > 0 else 0.0

            f1_score = (
                2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                if (avg_precision + avg_recall) > 0
                else 0.0
            )

            logger.info(
                f"Extraction F1: {f1_score:.2%} "
                f"(P={avg_precision:.2%}, R={avg_recall:.2%})"
            )

            return f1_score

        except Exception as e:
            logger.error(f"Extraction evaluation failed: {str(e)}")
            raise AppException(
                message="Failed to evaluate extraction",
                details={"error": str(e)},
            ) from e

    async def run_full_evaluation(
        self,
        db: AsyncSession,
    ) -> dict[str, float]:
        """Run full evaluation pipeline and save results."""
        try:
            logger.info("Starting full evaluation pipeline")

            rag_accuracy = await self.evaluate_rag(db)
            extraction_f1 = await self.evaluate_extraction()

            previous_results = self._load_previous_results()
            regression_delta = 0.0

            if previous_results:
                prev_rag = previous_results.get("rag_accuracy", 0.0)
                prev_f1 = previous_results.get("extraction_f1", 0.0)
                regression_delta = (
                    (rag_accuracy - prev_rag) + (extraction_f1 - prev_f1)
                ) / 2

            results = {
                "rag_accuracy": rag_accuracy,
                "extraction_f1": extraction_f1,
                "regression_delta": regression_delta,
                "timestamp": datetime.utcnow().isoformat(),
            }

            self._save_results(results)

            logger.info(
                f"Evaluation complete: RAG={rag_accuracy:.2%}, "
                f"F1={extraction_f1:.2%}, delta={regression_delta:+.2%}"
            )

            return results

        except Exception as e:
            logger.error(f"Full evaluation failed: {str(e)}")
            raise AppException(
                message="Failed to run evaluation",
                details={"error": str(e)},
            ) from e

    def _load_previous_results(self) -> dict | None:
        """Load the most recent evaluation results."""
        try:
            result_files = sorted(self.results_dir.glob("eval_*.json"), reverse=True)
            if result_files:
                with open(result_files[0]) as f:
                    return json.load(f)
            return None
        except Exception:
            return None

    def _save_results(self, results: dict) -> None:
        """Save evaluation results to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"eval_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results to {filename}")


evaluator = Evaluator()
