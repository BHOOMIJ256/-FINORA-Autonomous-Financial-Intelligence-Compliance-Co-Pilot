from loguru import logger
import sys


def setup_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - {message}",
        level="DEBUG",
    )
    logger.add(
        "logs/finora.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )
    return logger
