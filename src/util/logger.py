import sys
from loguru import logger

# Remove default handler (to avoid double printing)
logger.remove()

# Add Console Handler (Pretty colors, show File + Line Number)
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler
logger.add(
    "logs/app.log",
    rotation = "10MB",
    retention="10 days",
    level="DEBUG",
    compression="zip"
)

# Export the logger so other files can import it
__all__ = ["logger"]