from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from src.pose.pose import Pose
from src.util.logger import logger

@dataclass
class SearchFilters:
    """
    Defines strict criteria for a database search.
    Acts as a 'contract' between the Agent and the Engine.
    """
    fn_name: str
    is_person_search: bool
    semantic_query: Optional[str] = None
    people: List[str] = field(default_factory=list)
    year: Optional[int] = None
    pose: Optional[Pose] = None
    month: Optional[int] = None
    time_period: Optional[str] = None

    # For Generating
    prompt: Optional[str] = None

    @classmethod
    def from_openai_args(cls, fn_name: str, **kwargs) -> "SearchFilters":
        """
        Factory: Converts a raw OpenAI dictionary into a strict SearchFilters object.
        Safely handles string-to-Enum conversion errors.
        """
        # 1. Parse Pose
        pose_str = kwargs.get("pose")
        parsed_pose = None
        if pose_str:
            try:
                parsed_pose = Pose(pose_str)
            except ValueError:
                # Log warning or just ignore invalid enums
                logger.warning(f"⚠️ Warning: OpenAI returned invalid pose '{pose_str}'")
        else:
            # Only default to front if people filter is set
            if kwargs.get("people"):
                parsed_pose = Pose.FRONT
            else:
                parsed_pose = None

        # 3. Return the Clean Object
        return cls(
            fn_name=fn_name,
            people=kwargs.get("people", []),
            semantic_query=kwargs.get("search_query"),
            year=kwargs.get("year"),
            month=kwargs.get("month"),
            time_period=kwargs.get("time_period"),
            pose=parsed_pose,
            is_person_search=kwargs.get("requires_people", True) ,
            prompt=kwargs.get("prompt")
        )

    def __str__(self) -> str:
        """
        Returns a clean, human-readable summary of active filters.
        Example: "[People: Jacob | Pose: Side]"
        """
        parts = []

        # Only add fields that are NOT empty
        if self.people: parts.append(f"People: {', '.join(self.people)}")
        if self.pose: parts.append(f"Pose: {self.pose.value}")
        if self.year: parts.append(f"Year: {self.year}")
        if self.month: parts.append(f"Month: {self.month}")
        if self.time_period: parts.append(f"Time: {self.time_period}")
        if self.semantic_query: parts.append(f"Query: '{self.semantic_query}'")

        if not parts:
            return "[Filters: None]"

        return f"[Filters: {' | '.join(parts)}]"

    def to_dict(self):
        """
        Converts the object to a dictionary for st.json().
        Handles Enum serialization and removes empty fields.
        """
        data = {}
        for k, v in self.__dict__.items():
            # Skip empty fields for a cleaner UI
            if v is None or v == []:
                continue

            # CRITICAL: Convert Enums to string values (e.g., Pose.FRONT -> "Front")
            if isinstance(v, Enum):
                data[k] = v.value
            else:
                data[k] = v
        return data