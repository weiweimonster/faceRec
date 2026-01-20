"""
PersistStage: Save processed results to the database.
"""
from __future__ import annotations

from typing import List, Optional

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.db.storage import DatabaseManager
from src.util.logger import logger


class PersistStage(ProcessingStage):
    """
    Final stage: persist processed results to SQLite and ChromaDB.

    Supports both individual saves and batch commits for efficiency.
    """

    def __init__(self, db: DatabaseManager, batch_commit: bool = True):
        """
        Args:
            db: DatabaseManager instance
            batch_commit: Whether to use batch commits (more efficient)
        """
        self.db = db
        self.batch_commit = batch_commit

    @property
    def name(self) -> str:
        return "persist"

    @property
    def supports_gpu_batching(self) -> bool:
        return False  # I/O bound

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Persist all processable items to the database."""
        successful = 0
        failed = 0

        for item in items:
            if not item.is_processable():
                continue

            if not self._validate_item(item):
                item.mark_error("Validation failed: missing required fields")
                failed += 1
                continue

            try:
                photo_id = self.db.save_result(
                    result=item.result,
                    original_path=item.result.original_path,
                    display_path=item.result.display_path,
                    file_hash=item.file_hash,
                )
                item.result.photo_id = photo_id
                successful += 1
            except Exception as e:
                item.mark_error(f"Database save failed: {e}")
                failed += 1
                logger.error(f"Failed to save {item.result.original_path}: {e}")

        logger.info(f"Persist stage: {successful} saved, {failed} failed")
        return items

    def _validate_item(self, item: PipelineItem) -> bool:
        """Validate that item has required fields for persistence."""
        result = item.result

        if not result.original_path:
            return False
        if not result.display_path:
            return False
        if not item.file_hash:
            return False
        if result.semantic_vector is None:
            return False

        return True
