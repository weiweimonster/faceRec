from enum import Enum
from typing import Tuple

class Pose(Enum):
    # Standard Directions
    FRONT = "Front"
    SIDE_LEFT = "Side-Left"
    SIDE_RIGHT = "Side-Right"
    UP = "Up"
    DOWN = "Down"

    # Corners (Optional, but good for completeness)
    UP_LEFT = "Up-Left"
    UP_RIGHT = "Up-Right"
    DOWN_LEFT = "Down-Left"
    DOWN_RIGHT = "Down-Right"

    def __str__(self):
        return self.value

    @property
    def anchors(self) -> Tuple[float, float]:
        """Returns the (Yaw, Pitch) bullseye for this pose."""
        # Using the thresholds from your from_angles method as anchor points
        mapping = {
            Pose.FRONT: (0.0, 0.0),
            Pose.SIDE_LEFT: (45.0, 0.0),
            Pose.SIDE_RIGHT: (-45.0, 0.0),
            Pose.UP: (0.0, -30.0),
            Pose.DOWN: (0.0, 30.0),
            Pose.UP_LEFT: (35.0, -25.0),
            Pose.UP_RIGHT: (-35.0, -25.0),
            Pose.DOWN_LEFT: (35.0, 25.0),
            Pose.DOWN_RIGHT: (-35.0, 25.0),
        }
        return mapping.get(self, (0.0, 0.0))

    @classmethod
    def from_angles(cls, yaw: float, pitch: float) -> "Pose":
        """
        Factory method to convert raw 3D angles into a PoseDirection Enum.

        Args:
            yaw (float): Horizontal rotation (+Left / -Right)
            pitch (float): Vertical rotation (+Down / -Up)
        """
        # --- Thresholds (Adjust sensitivity here) ---
        H_THRESH = 25.0  # Degrees to trigger Side
        V_THRESH = 20.0  # Degrees to trigger Up/Down

        # 1. Determine Horizontal Component
        if yaw > H_THRESH:
            h_dir = "Left"
        elif yaw < -H_THRESH:
            h_dir = "Right"
        else:
            h_dir = "Center"

        # 2. Determine Vertical Component
        if pitch > V_THRESH:
            v_dir = "Down"
        elif pitch < -V_THRESH:
            v_dir = "Up"
        else:
            v_dir = "Center"

        # 3. Map to Enum
        if h_dir == "Center" and v_dir == "Center":
            return cls.FRONT

        if v_dir == "Center":
            # Pure Horizontal (Side-Left / Side-Right)
            return cls(f"Side-{h_dir}")

        if h_dir == "Center":
            # Pure Vertical (Up / Down)
            return cls(v_dir)

        # Combination (Up-Left, Down-Right, etc.)
        return cls(f"{v_dir}-{h_dir}")