from dataclasses import dataclass


@dataclass
class RecordSettings:
    """Settings for recording physics running."""
    video_directory: str = "test_video_dir"
    video_name: str = ""
    delete_at_init: bool = True
    fps: int = 24
    save_robot_view: bool = True
    generation_step: int = 9 # The number of generations between each recording
    
