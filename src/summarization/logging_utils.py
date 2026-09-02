from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> None:
    """Configure application console and file logging."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_logger = logging.getLogger(
        "summarization"
    )
    project_logger.setLevel(level)
    project_logger.propagate = False

    if project_logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path / "summarization.log",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    project_logger.addHandler(console_handler)
    project_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""

    return logging.getLogger(name)