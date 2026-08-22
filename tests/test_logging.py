import logging

from summarization.logging_utils import configure_logging, get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("test.module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_configure_logging_creates_log_directory(tmp_path):
    log_dir = tmp_path / "logs"

    configure_logging(log_dir=log_dir)

    assert log_dir.exists()
