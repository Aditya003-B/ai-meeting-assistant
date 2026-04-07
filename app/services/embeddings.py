"""Embedding generation service using OpenAI API."""

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""

    def __init__(self) -> None:
        """Initialize the embedding service with OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for the given text."""
        try:
            logger.info(f"Generating embedding for text of length {len(text)}")

            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )

            embedding = response.data[0].embedding

            logger.info(f"Embedding generated: {len(embedding)} dimensions")

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(
                message="Failed to generate embedding",
                details={"error": str(e), "text_length": len(text)},
            ) from e

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")

            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise EmbeddingError(
                message="Failed to generate batch embeddings",
                details={"error": str(e), "batch_size": len(texts)},
            ) from e


embedding_service = EmbeddingService()
