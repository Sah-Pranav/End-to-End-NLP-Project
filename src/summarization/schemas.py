from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """Request body for text summarization."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Text to summarize.",
    )


class SummarizationResponse(BaseModel):
    """Response returned after summarization."""

    summary: str
    model_name: str
    latency_ms: float
