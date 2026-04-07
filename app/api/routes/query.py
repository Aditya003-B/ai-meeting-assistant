"""Query API routes for RAG-based question answering."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.evaluator import evaluator
from app.agents.rag import rag_chain
from app.core.database import get_db
from app.models.schemas import EvalResult, ErrorResponse, QueryRequest, QueryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about meetings",
    description="Use RAG to answer questions based on historical meeting data",
    responses={
        500: {"model": ErrorResponse, "description": "RAG query failed"},
    },
)
async def query_meetings(
    query: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Answer a question using RAG over meeting history."""
    try:
        logger.info(f"Processing query: {query.question[:100]}...")

        answer, sources = await rag_chain.answer_question(
            db=db,
            question=query.question,
            top_k=query.top_k,
        )

        logger.info(f"Query answered with {len(sources)} sources")

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "RAGError",
                "message": "Failed to answer query",
                "details": {"error": str(e)},
            },
        ) from e


@router.post(
    "/eval/run",
    response_model=EvalResult,
    status_code=status.HTTP_200_OK,
    summary="Run offline evaluation",
    description="Trigger offline evaluation pipeline for RAG and extraction accuracy",
    responses={
        500: {"model": ErrorResponse, "description": "Evaluation failed"},
    },
)
async def run_evaluation(
    db: AsyncSession = Depends(get_db),
) -> EvalResult:
    """Run the offline evaluation pipeline."""
    try:
        logger.info("Starting offline evaluation")

        results = await evaluator.run_full_evaluation(db)

        logger.info(
            f"Evaluation complete: RAG={results['rag_accuracy']:.2%}, "
            f"F1={results['extraction_f1']:.2%}"
        )

        return EvalResult(
            rag_accuracy=results["rag_accuracy"],
            extraction_f1=results["extraction_f1"],
            regression_delta=results["regression_delta"],
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "EvaluationError",
                "message": "Failed to run evaluation",
                "details": {"error": str(e)},
            },
        ) from e
