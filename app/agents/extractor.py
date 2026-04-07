"""LangGraph agent for extracting meeting summaries, action items, and decisions."""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.core.exceptions import ExtractionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExtractorState(TypedDict):
    """State schema for the extractor agent."""

    transcript: str
    summary: str
    action_items: list[dict]
    decisions: list[dict]
    confidence: float
    requires_human_review: bool


class ExtractionAgent:
    """LangGraph agent for extracting structured information from meeting transcripts."""

    def __init__(self) -> None:
        """Initialize the extraction agent with LLM and graph."""
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ExtractorState)

        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("extract_actions", self._extract_actions_node)
        workflow.add_node("extract_decisions", self._extract_decisions_node)
        workflow.add_node("score_confidence", self._score_confidence_node)

        workflow.set_entry_point("summarize")
        workflow.add_edge("summarize", "extract_actions")
        workflow.add_edge("extract_actions", "extract_decisions")
        workflow.add_edge("extract_decisions", "score_confidence")
        workflow.add_edge("score_confidence", END)

        return workflow.compile()

    async def _summarize_node(self, state: ExtractorState) -> dict:
        """Summarize the meeting transcript in 3-5 sentences."""
        logger.info("Summarizing transcript")

        messages = [
            SystemMessage(
                content="You are an expert at summarizing meeting transcripts. "
                "Provide a concise summary in 3-5 sentences covering the main topics discussed."
            ),
            HumanMessage(content=f"Summarize this meeting transcript:\n\n{state['transcript']}"),
        ]

        response = await self.llm.ainvoke(messages)
        summary = response.content.strip()

        logger.info(f"Generated summary: {len(summary)} chars")

        return {"summary": summary}

    async def _extract_actions_node(self, state: ExtractorState) -> dict:
        """Extract action items with owner and deadline."""
        logger.info("Extracting action items")

        messages = [
            SystemMessage(
                content="You are an expert at extracting action items from meeting transcripts. "
                "Extract all action items with the person responsible (owner) and deadline if mentioned. "
                "Return a JSON array of objects with keys: owner, task, deadline (null if not mentioned). "
                "Return only the JSON array, no other text."
            ),
            HumanMessage(
                content=f"Extract action items from this transcript:\n\n{state['transcript']}"
            ),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content.strip()

        try:
            import json

            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            action_items = json.loads(content)

            if not isinstance(action_items, list):
                action_items = []

            logger.info(f"Extracted {len(action_items)} action items")

            return {"action_items": action_items}

        except json.JSONDecodeError:
            logger.warning("Failed to parse action items JSON, returning empty list")
            return {"action_items": []}

    async def _extract_decisions_node(self, state: ExtractorState) -> dict:
        """Extract key decisions made in the meeting."""
        logger.info("Extracting decisions")

        messages = [
            SystemMessage(
                content="You are an expert at extracting key decisions from meeting transcripts. "
                "Extract all important decisions made during the meeting. "
                "Return a JSON array of objects with keys: decision, context (optional). "
                "Return only the JSON array, no other text."
            ),
            HumanMessage(
                content=f"Extract decisions from this transcript:\n\n{state['transcript']}"
            ),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content.strip()

        try:
            import json

            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            decisions = json.loads(content)

            if not isinstance(decisions, list):
                decisions = []

            logger.info(f"Extracted {len(decisions)} decisions")

            return {"decisions": decisions}

        except json.JSONDecodeError:
            logger.warning("Failed to parse decisions JSON, returning empty list")
            return {"decisions": []}

    async def _score_confidence_node(self, state: ExtractorState) -> dict:
        """Score extraction confidence based on transcript clarity."""
        logger.info("Scoring extraction confidence")

        transcript_length = len(state["transcript"])
        action_items_count = len(state["action_items"])
        decisions_count = len(state["decisions"])

        confidence = 1.0

        if transcript_length < 100:
            confidence *= 0.5
        elif transcript_length < 300:
            confidence *= 0.7

        if action_items_count == 0 and decisions_count == 0:
            confidence *= 0.6

        has_owners = all(
            item.get("owner") for item in state["action_items"] if isinstance(item, dict)
        )
        if action_items_count > 0 and not has_owners:
            confidence *= 0.8

        requires_review = confidence < settings.confidence_threshold

        logger.info(
            f"Confidence score: {confidence:.2f}, requires_review: {requires_review}"
        )

        return {
            "confidence": confidence,
            "requires_human_review": requires_review,
        }

    async def extract(self, transcript: str) -> ExtractorState:
        """Run the extraction agent on a transcript."""
        try:
            logger.info(f"Starting extraction for transcript of length {len(transcript)}")

            initial_state: ExtractorState = {
                "transcript": transcript,
                "summary": "",
                "action_items": [],
                "decisions": [],
                "confidence": 0.0,
                "requires_human_review": False,
            }

            result = await self.graph.ainvoke(initial_state)

            logger.info(
                f"Extraction complete: confidence={result['confidence']:.2f}, "
                f"action_items={len(result['action_items'])}, "
                f"decisions={len(result['decisions'])}"
            )

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise ExtractionError(
                message="Failed to extract meeting information",
                details={"error": str(e)},
            ) from e


extraction_agent = ExtractionAgent()
