from typing import Any, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict


_ContentType = TypeVar('_ContentType')


class PromptTokensDetails(BaseModel):
    audio_tokens: Optional[int] = Field(None, description="Number of audio tokens in the prompt.")
    cached_tokens: Optional[int] = Field(None, description="Number of cached tokens in the prompt.")
    text_tokens: Optional[int] = Field(None, description="Number of text tokens in the prompt.")
    image_tokens: Optional[int] = Field(None, description="Number of image tokens in the prompt.")

    model_config = ConfigDict(extra="ignore")


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: Optional[int] = Field(None, description="Number of accepted prediction tokens in the completion.")
    audio_tokens: Optional[int] = Field(None, description="Number of audio tokens in the completion.")
    reasoning_tokens: Optional[int] = Field(None, description="Number of reasoning tokens in the completion.")
    rejected_prediction_tokens: Optional[int] = Field(None, description="Number of rejected prediction tokens in the completion.")
    text_tokens: Optional[int] = Field(None, description="Number of text tokens in the completion.")

    model_config = ConfigDict(extra="ignore")


class UsageMetrics(BaseModel):
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the generated completion.")
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt.")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used in the request (prompt + completion).")
    completion_tokens_details: Optional[CompletionTokensDetails] = Field(None, description="Detailed breakdown of completion tokens.")
    prompt_tokens_details: Optional[PromptTokensDetails] = Field(None, description="Detailed breakdown of prompt tokens.")

    model_config = ConfigDict(extra="ignore")


class ThinagentResponse(BaseModel, Generic[_ContentType]):
    content: _ContentType
    content_type: str = Field(description="Indicates the name of the Pydantic model in 'content' or 'str'.")
    response_id: Optional[str] = Field(None, description="Unique identifier for the LLM response.")
    created_timestamp: Optional[int] = Field(None, description="Timestamp of when the response was created by the LLM.")
    model_used: Optional[str] = Field(None, description="The model that generated this response.")
    finish_reason: Optional[str] = Field(None, description="The reason the model stopped generating tokens (e.g., 'stop', 'tool_calls').")
    metrics: Optional[UsageMetrics] = Field(None, description="Token usage statistics and details for the request.")
    system_fingerprint: Optional[str] = Field(None, description="A system fingerprint representing the backend configuration that generated the response.")
    artifact: Optional[Any] = Field(None, description="For any additional data from the LLM provider not covered by other fields. Defaults to None.")
    tool_name: Optional[str] = Field(None, description="Name of the tool associated with this response chunk, if applicable.")
    tool_call_id: Optional[str] = Field(None, description="Unique identifier of the tool call that produced this chunk, if applicable.")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")


class ThinagentResponseStream(ThinagentResponse[_ContentType], Generic[_ContentType]):
    stream_options: Optional[Any] = Field(None, description="Streaming options for this chunk. Defaults to None.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def __repr__(self) -> str:
        data = self.model_dump()
        field_str = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"{self.__class__.__name__}({field_str})"

    def __str__(self) -> str:
        return self.__repr__() 