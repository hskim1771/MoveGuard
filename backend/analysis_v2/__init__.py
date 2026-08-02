"""MoveGuard posture-analysis v2.

The package is intentionally isolated from the legacy endpoints so that pose,
hand and object models can be replaced without changing the RULA engine or the
public response contract.
"""

from .pipeline import PostureAnalysisPipeline, get_pipeline

__all__ = ["PostureAnalysisPipeline", "get_pipeline"]
