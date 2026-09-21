"""VideoGenerator protocol: the provider-independent video-generation boundary.

The content pipeline (stage runner + validator + CLI) depends on this protocol,
never on the mock implementation. A future Colab-hosted / open-source GPU worker
implements the same `generate(...)` signature behind this boundary, and the
Stage-7 stage runner, validator, resume/retry logic, and CLI change only by
swapping the generator instance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.scene_planner.models import ProductionPlan
from agents.video_generation.models import (
    VideoGenerationConfig,
    VideoGenerationResult,
)


@runtime_checkable
class VideoGenerator(Protocol):
    """Provider-independent video generation interface.

    `plan` is the Stage-6 ProductionPlan (the stable CONTENT -> VIDEO boundary).
    `previous` (when given) is the prior VideoGenerationResult for the SAME plan,
    enabling resume/retry: completed scenes are preserved, failed retryable
    scenes may be retried, pending scenes are processed, and non-retryable
    failures stay recorded. Implementations are deterministic and return a
    validated `VideoGenerationResult`.
    """

    def generate(
        self,
        plan: ProductionPlan,
        config: VideoGenerationConfig | None = None,
        previous: VideoGenerationResult | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> VideoGenerationResult:
        ...