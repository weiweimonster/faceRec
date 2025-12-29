from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class Pose(Enum):
    FRONT = "Front"
    SIDE = "Side"
    SIDE_LEFT = "Side-Left"
    SIDE_RIGHT = "Side-Right"

class Emotion(Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SURPRISE = "surprise"
    SAD = "sad"


@dataclass
class SearchFilters:
    """
    Defines strict criteria for a database search.
    Acts as a 'contract' between the Agent and the Engine.
    """
    semantic_query: Optional[str] = None
    people: List[str] = field(default_factory=list)
    year: Optional[int] = None
    pose: Optional[Pose] = None
    emotion: Optional[Emotion] = None


    # You can also move quality settings here if you want strict overrides
    min_blur: Optional[float] = None

    @classmethod
    def from_openai_args(cls, args: Dict[str, Any]) -> "SearchFilters":
        """
        Factory: Converts a raw OpenAI dictionary into a strict SearchFilters object.
        Safely handles string-to-Enum conversion errors.
        """
        # 1. Parse Pose
        pose_str = args.get("pose")
        parsed_pose = None
        if pose_str:
            try:
                parsed_pose = Pose(pose_str)
            except ValueError:
                # Log warning or just ignore invalid enums
                print(f"⚠️ Warning: OpenAI returned invalid pose '{pose_str}'")
        else:
            parsed_pose = Pose.FRONT

        # 2. Parse Emotion
        emotion_str = args.get("emotion")
        parsed_emotion = None
        if emotion_str:
            try:
                parsed_emotion = Emotion(emotion_str)
            except ValueError:
                print(f"⚠️ Warning: OpenAI returned invalid emotion '{emotion_str}'")

        # 3. Return the Clean Object
        return cls(
            people=args.get("people", []),
            semantic_query=args.get("search_query"),
            year=args.get("year"),
            pose=parsed_pose,
            emotion=parsed_emotion
        )

    def __str__(self) -> str:
        """
        Returns a clean, human-readable summary of active filters.
        Example: "[People: Jacob | Pose: Side]"
        """
        parts = []

        # Only add fields that are NOT empty
        if self.people:
            parts.append(f"People: {', '.join(self.people)}")

        if self.pose:
            parts.append(f"Pose: {self.pose.value}")  # .value prints "Side" instead of Pose.SIDE

        if self.emotion:
            parts.append(f"Emotion: {self.emotion.value}")

        if self.year:
            parts.append(f"Year: {self.year}")

        if self.semantic_query:
            parts.append(f"Query: '{self.semantic_query}'")

        if not parts:
            return "[Filters: None]"

        return f"[Filters: {' | '.join(parts)}]"

    def is_empty(self):
        return not (self.people or self.year or self.pose or self.emotion)