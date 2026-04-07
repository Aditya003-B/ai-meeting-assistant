"""RAG chain for answering questions over meeting history."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import RAGError
from app.models.schemas import QuerySource
from app.services.vector_store import vector_store_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGChain:
    """RAG chain for question answering over meeting notes."""

    def __init__(self) -> None:
        """Initialize the RAG chain with LLM."""
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )

    async def answer_question(
        self,
        db: AsyncSession,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, list[QuerySource]]:
        """Answer a question using RAG over meeting history."""
        try:
            logger.info(f"Answering question: {question[:100]}...")

            similar_meetings = await vector_store_service.similarity_search(
                db=db,
                query_text=question,
                top_k=top_k,
            )

            if not similar_meetings:
                return (
                    "I couldn't find any relevant meetings to answer your question.",
                    [],
                )

            context_parts = []
            sources: list[QuerySource] = []

            for meeting, score in similar_meetings:
                context_parts.append(
                    f"Meeting: {meeting.title}\n"
                    f"Summary: {meeting.summary or 'No summary available'}\n"
                    f"Transcript excerpt: {meeting.transcript[:500]}...\n"
                )

                sources.append(
                    QuerySource(
                        meeting_id=meeting.id,
                        title=meeting.title,
                        relevance_score=score,
                    )
                )

            context = "\n\n---\n\n".join(context_parts)

            system_prompt = (
                "You are a helpful assistant that answers questions about past meetings. "
                "Use the provided meeting context to answer the user's question accurately. "
                "If the context doesn't contain enough information to answer the question, "
                "say so clearly. Always cite which meeting(s) you're referencing in your answer."
            )

            user_prompt = (
                f"Context from relevant meetings:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Please provide a comprehensive answer based on the meeting context above."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            answer = response.content.strip()

            logger.info(
                f"Generated answer of length {len(answer)} using {len(sources)} sources"
            )

            return answer, sources

        except Exception as e:
            logger.error(f"RAG question answering failed: {str(e)}")
            raise RAGError(
                message="Failed to answer question",
                details={"error": str(e), "question": question},
            ) from e


rag_chain = RAGChain()
