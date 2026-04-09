"""
Logging configuration.

Call configure_logging(settings) once at application startup (in main.py
and api/app.py). Every module can then simply do:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")

WHY this file exists
--------------------
The original code used:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

That format drops the timestamp, log level, and module name — making logs
almost impossible to diagnose in production. It also cannot be switched to
JSON, which is required for AWS CloudWatch Insights queries.

Two output modes
----------------
text  (LOG_FORMAT=text)  — human-readable, good for local development
json  (LOG_FORMAT=json)  — structured JSON, required for CloudWatch

In Docker/ECS the Dockerfile sets LOG_FORMAT=json automatically.
You can override with LOG_FORMAT=text locally in your .env file.

Paddle noise suppression
------------------------
PaddlePaddle prints a lot at DEBUG/INFO level (model loading progress,
inference timings, etc.). We silence those loggers to WARNING so they
do not drown out application logs.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str = "INFO", log_format: str = "text") -> None:
    """
    Set up root logger with the requested level and format.

    Args:
        log_level:  One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        log_format: "text" for human-readable, "json" for structured JSON.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        # python-json-logger emits one JSON object per log line.
        # CloudWatch Logs Insights can then query fields like `levelname`,
        # `name`, `message` directly without regex parsing.
        try:
            from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]

            formatter = JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
        except ImportError:
            # python-json-logger not installed — fall back to text and warn.
            formatter = _text_formatter()
            logging.warning(
                "python-json-logger is not installed. "
                "Set LOG_FORMAT=text or run: pip install python-json-logger"
            )
    else:
        formatter = _text_formatter()

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers that were added before (e.g. by basicConfig calls
    # from imported libraries) so we do not get duplicate log lines.
    root.handlers.clear()
    root.addHandler(handler)

    # Paddle and PIL print model-loading progress at INFO; silence to WARNING.
    for noisy in ("ppdet", "paddle", "paddleocr", "paddlex", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _text_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
