class AnalysisError(RuntimeError):
    """Base exception for posture-analysis v2."""


class InvalidAnalysisInput(AnalysisError):
    """The supplied image or landmark payload cannot be analyzed."""


class EstimatorUnavailable(AnalysisError):
    """A configured model runtime or model asset is unavailable."""
