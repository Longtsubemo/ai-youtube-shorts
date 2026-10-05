"""Stage 5: Video Generation.

Reads ``scene_plan.json`` and writes ``video.mp4``.

CLI: ``python -m agents.video_generation.video_cli --run-id <run_id>``
"""

from agents.video_generation.stage_simple import write_video_placeholder

__all__ = ["write_video_placeholder"]