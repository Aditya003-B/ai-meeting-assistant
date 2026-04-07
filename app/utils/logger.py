"""Structured logging configuration for the application."""

import logging
import sys
from typing import Any

from app.core.config import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured context."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        parts = [f"{k}={v}" for k, v in log_data.items()]
        return " | ".join(parts)


def setup_logger(name: str) -> logging.Logger:
    """Set up a structured logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return setup_logger(name)
