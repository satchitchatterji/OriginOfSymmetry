from dataclasses import dataclass


@dataclass
class RecordSettings:
    """Settings for recording physics running."""

    video_directory: str
    fps: int = 24
    save_robot_view: bool = True
    generation_step: int = 1 # The number of generations between each recording
    
