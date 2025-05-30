"""
Module implementing the Agent class for orchestrating LLM interactions and tool execution.
"""

import json
import logging
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Iterator, AsyncIterator, TypeVar, Generic, cast, overload, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import litellm
from litellm import completion as litellm_completion
from pydantic import BaseModel, ValidationError # type: ignore
from thinagents.core.tool import ThinAgentsTool, tool as tool_decorator
from thinagents.memory import BaseMemory, ConversationInfo
from thinagents.utils.prompts import PromptConfig
from thinagents.core.response_models import (
    ThinagentResponse,
    ThinagentResponseStream,
    UsageMetrics,
    CompletionTokensDetails,
    PromptTokensDetails,
)

logger = logging.getLogger(__name__)

_ExpectedContentType = TypeVar('_ExpectedContentType', bound=BaseModel)

DEFAULT_MAX_STEPS = 15
DEFAULT_TOOL_TIMEOUT = 30.0
MAX_JSON_CORRECTION_ATTEMPTS = 3


class AgentError(Exception):
    """Base exception for Agent-related errors."""
    pass


class ToolExecutionError(AgentError):
    """Exception raised when tool execution fails."""
    pass


class MaxStepsExceededError(AgentError):
    """Exception raised when max steps are exceeded."""
    pass


def generate_tool_schemas(
    tools: Union[List[ThinAgentsTool], List[Callable]],
) -> Tuple[List[Dict], Dict[str, ThinAgentsTool]]:
    """
    Generate JSON schemas for provided tools and return tool schemas list and tool maps.

    Args:
        tools: A list of ThinAgentsTool instances or callables decorated with @tool.

    Returns:
        Tuple[List[Dict], Dict[str, ThinAgentsTool]]: A list of tool schema dictionaries and a mapping from tool names to ThinAgentsTool instances.
        
    Raises:
        AgentError: If tool schema generation fails.
    """
    tool_schemas = []
    tool_maps: Dict[str, ThinAgentsTool] = {}

    for tool in tools:
        try:
            if isinstance(tool, ThinAgentsTool):
                schema_data = tool.tool_schema()
                tool_maps[tool.__name__] = tool
            else:
                _tool = tool_decorator(tool)
                schema_data = _tool.tool_schema()
                tool_maps[_tool.__name__] = _tool
            
            # extract the actual OpenAI tool schema from our wrapper format
            if isinstance(schema_data, dict) and "tool_schema" in schema_data:
                # new format with return_type metadata
                actual_schema = schema_data["tool_schema"]
            else:
                # legacy format - direct schema
                actual_schema = schema_data
            
            tool_schemas.append(actual_schema)
        except Exception as e:
            logger.error(f"Failed to generate schema for tool {tool}: {e}")
            raise AgentError(f"Tool schema generation failed for {tool}: {e}") from e

    return tool_schemas, tool_maps


def _validate_agent_config(
    name: str,
    model: str,
    max_steps: int,
) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("Agent name must be a non-empty string")
    
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string")
    
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")


class Agent(Generic[_ExpectedContentType]):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        tools: Optional[Union[List[ThinAgentsTool], List[Callable]]] = None,
        sub_agents: Optional[List["Agent"]] = None,
        prompt: Optional[Union[str, PromptConfig]] = None,
        instructions: Optional[List[str]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        concurrent_tool_execution: bool = True,
        response_format: Optional[Type[_ExpectedContentType]] = None,
        enable_schema_validation: bool = True,
        description: Optional[str] = None,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        memory: Optional[BaseMemory] = None,
        **kwargs,
    ):
        """
        Initializes an instance of the Agent class.

        Args:
            name: The name of the agent.
            model: The identifier of the language model to be used by the agent (e.g., "gpt-3.5-turbo").
            api_key: Optional API key for authenticating with the model's provider.
            api_base: Optional base URL for the API, if using a custom or self-hosted model.
            api_version: Optional API version, required by some providers like Azure OpenAI.
            tools: A list of tools that the agent can use.
                Tools can be instances of `ThinAgentsTool` or callable functions decorated with `@tool`.
            sub_agents: A list of `Agent` instances that should be exposed as tools to this
                parent agent. Each sub-agent will be wrapped in a ThinAgents tool that takes a
                single string parameter named `input` and returns the sub-agent's response. This
                allows the parent agent to delegate work to specialised child agents.
            prompt: The system prompt to guide the agent's behavior.
                This can be a simple string or a `PromptConfig` object for more complex prompt engineering.
            instructions: A list of additional instruction strings to be appended to the system prompt.
                Ignored when `prompt` is an instance of `PromptConfig`.
            max_steps: The maximum number of conversational turns or tool execution
                sequences the agent will perform before stopping. Defaults to 15.
            parallel_tool_calls: If True, allows the agent to request multiple tool calls
                in a single step from the language model. Defaults to False.
            concurrent_tool_execution: If True and `parallel_tool_calls` is also True,
                the agent will execute multiple tool calls concurrently using a thread pool.
                Defaults to True.
            response_format: Configuration for enabling structured output from the model.
                This should be a Pydantic model.
            enable_schema_validation: If True, enables schema validation for the response format.
                Defaults to True.
            description: Optional description for the agent.
            tool_timeout: Timeout in seconds for tool execution. Defaults to 30.0.
            memory: Optional BaseMemory instance for storing conversation history and context.
                When provided, the agent will automatically manage conversation history across
                multiple run() calls using conversation_id parameter. Use InMemoryStore with
                store_tool_artifacts=True to also store tool artifacts alongside conversation history.
            **kwargs: Additional keyword arguments that will be passed directly to the `litellm.completion` function.
        """
        _validate_agent_config(name, model, max_steps)

        self.name = name
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.max_steps = max_steps
        self.prompt = prompt
        self.instructions = instructions or []
        self.sub_agents = sub_agents or []
        self.description = description
        self.tool_timeout = tool_timeout

        self.response_format_model_type = response_format
        self.enable_schema_validation = enable_schema_validation
        if self.response_format_model_type:
            litellm.enable_json_schema_validation = self.enable_schema_validation

        self.concurrent_tool_execution = concurrent_tool_execution
        self.kwargs = kwargs

        self._provided_tools = tools or []

        self.granular_stream = True
        """Whether to emit per-character ThinagentResponseStream chunks when streaming"""

        # Initialize tools and sub-agents
        self._initialize_tools()

        # Initialize memory-related attributes
        self.memory = memory

    def _initialize_tools(self) -> None:
        """Initialize tools and sub-agents."""
        try:
            sub_agent_tools: List[ThinAgentsTool] = [
                self._make_sub_agent_tool(sa) for sa in self.sub_agents
            ]
            combined_tools = (self._provided_tools or []) + sub_agent_tools
            self.tool_schemas, self.tool_maps = generate_tool_schemas(combined_tools)
            logger.info(f"Initialized {len(self.tool_maps)} tools for agent '{self.name}'")
        except Exception as e:
            logger.error(f"Failed to initialize tools for agent '{self.name}': {e}")
            raise AgentError(f"Tool initialization failed: {e}") from e

    def _make_sub_agent_tool(self, sa: "Agent") -> ThinAgentsTool:
        """Create a ThinAgents tool that delegates calls to a sub-agent."""
        safe_name = sa.name.lower().strip().replace(" ", "_")

        def _delegate_to_sub_agent(input: str) -> Any:
            """Delegate input to the sub-agent."""
            try:
                return sa.run(input)
            except Exception as e:
                logger.error(f"Sub-agent '{sa.name}' execution failed: {e}")
                raise ToolExecutionError(f"Sub-agent execution failed: {e}") from e

        _delegate_to_sub_agent.__name__ = f"subagent_{safe_name}"
        _delegate_to_sub_agent.__doc__ = sa.description or (
            f"Forward the input to the '{sa.name}' sub-agent and return its response."
        )

        return tool_decorator(_delegate_to_sub_agent)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """
        Executes a tool by name with the provided arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        tool = self.tool_maps.get(tool_name)
        if tool is None:
            raise ToolExecutionError(f"Tool '{tool_name}' not found.")

        try:
            logger.debug(f"Executing tool '{tool_name}' with args: {tool_args}")

            if self.concurrent_tool_execution:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(tool, **tool_args)
                    try:
                        result = future.result(timeout=self.tool_timeout)
                        logger.debug(f"Tool '{tool_name}' executed successfully")
                        return result
                    except TimeoutError as e:
                        logger.error(f"Tool '{tool_name}' execution timed out after {self.tool_timeout}s")
                        raise ToolExecutionError(f"Tool '{tool_name}' execution timed out") from e
            else:
                result = tool(**tool_args)
                logger.debug(f"Tool '{tool_name}' executed successfully")
                return result

        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            raise ToolExecutionError(f"Tool '{tool_name}' execution failed: {e}") from e

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        if isinstance(self.prompt, PromptConfig):
            return self.prompt.add_instruction(f"Your name is {self.name}").build()
        base_prompt = (
            f"You are a helpful assistant. Answer the user's question to the best of your ability. Your name is {self.name}."
            if self.prompt is None
            else self.prompt
        )
        if self.instructions:
            base_prompt = f"{base_prompt}\n" + "\n".join(self.instructions)

        return base_prompt

    def _extract_usage_metrics(self, response: Any) -> Optional[UsageMetrics]:
        """Extract usage metrics from LLM response."""
        try:
            raw_usage = getattr(response, "usage", None)
            if not raw_usage:
                return None

            ct_details_data = getattr(raw_usage, "completion_tokens_details", None)
            pt_details_data = getattr(raw_usage, "prompt_tokens_details", None)

            ct_details = None
            if ct_details_data:
                ct_details = CompletionTokensDetails(
                    accepted_prediction_tokens=getattr(ct_details_data, "accepted_prediction_tokens", None),
                    audio_tokens=getattr(ct_details_data, "audio_tokens", None),
                    reasoning_tokens=getattr(ct_details_data, "reasoning_tokens", None),
                    rejected_prediction_tokens=getattr(ct_details_data, "rejected_prediction_tokens", None),
                    text_tokens=getattr(ct_details_data, "text_tokens", None),
                )

            pt_details = None
            if pt_details_data:
                pt_details = PromptTokensDetails(
                    audio_tokens=getattr(pt_details_data, "audio_tokens", None),
                    cached_tokens=getattr(pt_details_data, "cached_tokens", None),
                    text_tokens=getattr(pt_details_data, "text_tokens", None),
                    image_tokens=getattr(pt_details_data, "image_tokens", None),
                )
            
            return UsageMetrics(
                completion_tokens=getattr(raw_usage, "completion_tokens", None),
                prompt_tokens=getattr(raw_usage, "prompt_tokens", None),
                total_tokens=getattr(raw_usage, "total_tokens", None),
                completion_tokens_details=ct_details,
                prompt_tokens_details=pt_details,
            )
        except Exception as e:
            logger.warning(f"Failed to extract usage metrics: {e}")
            return None

    def _handle_json_correction(
        self, 
        messages: List[Dict], 
        raw_content: str, 
        error: Exception,
        attempt: int
    ) -> bool:
        """
        Handle JSON correction for structured output.
        
        Returns:
            True if correction should be attempted, False if max attempts reached
        """
        if attempt >= MAX_JSON_CORRECTION_ATTEMPTS:
            logger.error(f"Max JSON correction attempts ({MAX_JSON_CORRECTION_ATTEMPTS}) reached")
            return False
            
        logger.warning(f"JSON validation failed (attempt {attempt + 1}): {error}")
        
        schema_info = "unknown schema"
        if self.response_format_model_type:
            try:
                schema_dict = self.response_format_model_type.model_json_schema()
                schema_info = json.dumps(schema_dict) if isinstance(schema_dict, dict) else str(schema_dict)
            except Exception:
                schema_info = str(self.response_format_model_type)
        
        correction_prompt = (
            f"The JSON is invalid: {error}. Please fix the JSON and return it. "
            f"Returned JSON: {raw_content}, "
            f"Expected JSON schema: {schema_info}"
        )
        
        messages.append({"role": "user", "content": correction_prompt})
        return True

    def _serialize_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert tool call objects to JSON-serializable dictionaries.
        
        Args:
            tool_calls: List of tool call objects from LLM response
            
        Returns:
            List of serializable tool call dictionaries
        """
        serialized_calls = []
        
        for tc in tool_calls:
            try:
                # Handle both modern tool_calls format and legacy function_call
                if hasattr(tc, 'function') and hasattr(tc, 'id'):
                    # Modern tool_calls format
                    serialized_call = {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                elif hasattr(tc, 'name') and hasattr(tc, 'arguments'):
                    # Legacy function_call format
                    serialized_call = {
                        "id": getattr(tc, "id", f"call_{tc.name}"),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    }
                else:
                    # Fallback - try to extract what we can
                    serialized_call = {
                        "id": getattr(tc, "id", f"call_{getattr(tc, 'name', 'unknown')}"),
                        "type": "function",
                        "function": {
                            "name": getattr(tc, "name", "unknown"),
                            "arguments": getattr(tc, "arguments", "{}")
                        }
                    }
                
                serialized_calls.append(serialized_call)
                
            except Exception as e:
                logger.warning(f"Failed to serialize tool call {tc}: {e}")
                # Add a fallback entry to maintain structure
                serialized_calls.append({
                    "id": "unknown_call",
                    "type": "function", 
                    "function": {
                        "name": "unknown",
                        "arguments": "{}"
                    }
                })
        
        return serialized_calls

    def _process_tool_call_result(self, tool_call_result: Any) -> str:
        """Process tool call result and convert to string for LLM."""
        try:
            if isinstance(tool_call_result, ThinagentResponse):
                # Result from a sub-agent
                sub_agent_content_data = tool_call_result.content
                if isinstance(sub_agent_content_data, BaseModel): 
                    return sub_agent_content_data.model_dump_json()
                elif isinstance(sub_agent_content_data, str):
                    return sub_agent_content_data
                else:
                    return json.dumps(sub_agent_content_data)
            elif isinstance(tool_call_result, BaseModel):
                return tool_call_result.model_dump_json()
            elif isinstance(tool_call_result, str):
                return tool_call_result
            else:
                return json.dumps(tool_call_result)
        except Exception as e:
            logger.warning(f"Failed to serialize tool result: {e}")
            return str(tool_call_result)

    def _execute_single_tool_call(self, tc: Any) -> Dict[str, Any]:
        # sourcery skip: extract-method
        """Parse, execute, and format a single tool call."""
        tool_call_name = tc.function.name
        tool_call_id = tc.id
        tool = self.tool_maps.get(tool_call_name)
        return_type = getattr(tool, "return_type", "content")
        try:
            tool_call_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tool arguments for {tool_call_name} (ID: {tool_call_id}): {e}")
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_call_name,
                "content": json.dumps({
                    "error": str(e),
                    "message": "Failed to parse arguments",
                }),
            }

        try:
            raw_result = self._execute_tool(tool_call_name, tool_call_args)
            if return_type == "content_and_artifact" and isinstance(raw_result, tuple) and len(raw_result) == 2:
                content_value, artifact = raw_result
                self._tool_artifacts[tool_call_name] = artifact
            else:
                content_value = raw_result
                artifact = None

            content_for_llm = self._process_tool_call_result(content_value)
            tool_message = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_call_name,
                "content": content_for_llm,
            }
            if self._should_include_artifacts_in_messages() and artifact is not None:
                tool_message["artifact"] = artifact
            return tool_message
        except ToolExecutionError as e:
            logger.error(f"Tool execution error for {tool_call_name} (ID: {tool_call_id}): {e}")
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_call_name,
                "content": json.dumps({
                    "error": str(e),
                    "message": "Tool execution failed",
                }),
            }

    @overload
    def run(
        self,
        input: str,
        stream: Literal[False] = False,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> ThinagentResponse[_ExpectedContentType]:
        ...
    @overload
    def run(
        self,
        input: str,
        stream: Literal[True],
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Iterator[ThinagentResponseStream[Any]]:
        ...
    def run(
        self,
        input: str,
        stream: bool = False,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Any:
        """
        Run the agent with the given input and manage interactions with the language model and tools.

        Args:
            input: The user's input message to the agent.
            stream: If True, returns a stream of responses instead of a single response.
            stream_intermediate_steps: If True and stream=True, also stream intermediate tool calls and results.
            conversation_id: Optional conversation ID for memory retrieval

        Returns:
            ThinagentResponse[_ExpectedContentType] when stream=False, or Iterator[ThinagentResponseStream] when stream=True.
            
        Raises:
            AgentError: If agent execution fails
            MaxStepsExceededError: If max steps are exceeded
        """
        if not input or not isinstance(input, str):
            raise ValueError("Input must be a non-empty string")
            
        logger.info(f"Agent '{self.name}' starting execution with input length: {len(input)}")
        
        # Handle streaming response
        if stream:
            if self.response_format_model_type:
                raise ValueError("Streaming is not supported when response_format is specified.")
            return self._run_stream(input, stream_intermediate_steps, conversation_id)

        try:
            return self._run_sync(input, conversation_id)
        except Exception as e:
            logger.error(f"Agent '{self.name}' execution failed: {e}")
            if isinstance(e, (AgentError, MaxStepsExceededError)):
                raise
            raise AgentError(f"Agent execution failed: {e}") from e

    def _run_loop(self, messages: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> ThinagentResponse[_ExpectedContentType]:
        """Shared synchronous step loop."""
        steps = 0
        json_correction_attempts = 0
        while steps < self.max_steps:
            try:
                response = litellm_completion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    tools=self.tool_schemas,
                    response_format=self.response_format_model_type,
                    **self.kwargs,
                )
            except Exception as e:
                logger.error(f"LLM completion failed: {e}")
                raise AgentError(f"LLM completion failed: {e}") from e

            response_id = getattr(response, "id", None)
            created_timestamp = getattr(response, "created", None)
            model_used = getattr(response, "model", None)
            system_fingerprint = getattr(response, "system_fingerprint", None)
            metrics = self._extract_usage_metrics(response)

            try:
                if not hasattr(response, 'choices') or not response.choices:  # type: ignore
                    logger.error("Response has no choices")
                    raise AgentError("Invalid response structure: no choices")
                finish_reason = response.choices[0].finish_reason  # type: ignore
                message = response.choices[0].message  # type: ignore
                tool_calls = getattr(message, "tool_calls", None) or []
            except (IndexError, AttributeError) as e:
                logger.error(f"Invalid response structure: {e}")
                raise AgentError(f"Invalid response structure: {e}") from e

            if finish_reason == "stop" and not tool_calls:
                return self._handle_completion(
                    message, response_id, created_timestamp, model_used,
                    finish_reason, metrics, system_fingerprint, messages,
                    json_correction_attempts, conversation_id
                )
            if finish_reason == "tool_calls" or tool_calls:
                self._handle_tool_calls(tool_calls, message, messages, conversation_id)
                steps += 1
                continue

            steps += 1

        logger.warning(f"Agent '{self.name}' reached max steps ({self.max_steps})")
        raise MaxStepsExceededError(f"Max steps ({self.max_steps}) reached without final answer.")

    def _run_sync(self, input: str, conversation_id: Optional[str] = None) -> ThinagentResponse[_ExpectedContentType]:
        """Synchronous execution of the agent."""
        self._tool_artifacts: dict[str, Any] = {}  # initialize storage for tool artifacts
        messages = self._build_messages_with_memory(input, conversation_id)
        return self._run_loop(messages, conversation_id)

    def _handle_completion(
        self, 
        message: Any, 
        response_id: Optional[str],
        created_timestamp: Optional[int],
        model_used: Optional[str],
        finish_reason: Optional[str],
        metrics: Optional[UsageMetrics],
        system_fingerprint: Optional[str],
        messages: List[Dict],
        json_correction_attempts: int,
        conversation_id: Optional[str] = None,
    ) -> ThinagentResponse[_ExpectedContentType]:
        """Handle completion response without tool calls."""
        raw_content_from_llm = message.content

        if self.response_format_model_type:
            try:
                parsed_model = self.response_format_model_type.model_validate_json(raw_content_from_llm)
                final_content = cast(_ExpectedContentType, parsed_model)
                content_type_to_return = self.response_format_model_type.__name__
            except (ValidationError, json.JSONDecodeError) as e:
                if self._handle_json_correction(messages, raw_content_from_llm, e, json_correction_attempts):
                    return self._run_sync(messages[-1]["content"], conversation_id)
                # Max attempts reached, return error
                final_content = cast(_ExpectedContentType, f"JSON validation failed after {MAX_JSON_CORRECTION_ATTEMPTS} attempts: {e}")
                content_type_to_return = "str"
        else:
            final_content = cast(_ExpectedContentType, raw_content_from_llm)
            content_type_to_return = "str"

        assistant_response_message = {"role": "assistant", "content": raw_content_from_llm}
        messages.append(assistant_response_message)

        if conversation_id is not None and self.memory is not None:
            try:
                self._save_messages_to_memory(messages, conversation_id)
            except Exception as e:
                logger.warning(f"An unexpected error occurred when calling _save_messages_to_memory from _handle_completion for conversation '{conversation_id}': {e}")

        return ThinagentResponse(
            content=final_content,
            content_type=content_type_to_return,
            response_id=response_id,
            created_timestamp=created_timestamp,
            model_used=model_used,
            finish_reason=finish_reason,
            metrics=metrics,
            system_fingerprint=system_fingerprint,
            artifact=self._tool_artifacts,
            tool_name=None,
            tool_call_id=None,
        )

    def _handle_tool_calls(self, tool_calls: List[Any], message: Any, messages: List[Dict], conversation_id: Optional[str] = None) -> None:
        """Handle tool calls execution."""
        tool_call_outputs = []

        if tool_calls:
            if self.concurrent_tool_execution and len(tool_calls) > 1:
                with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
                    futures = {
                        executor.submit(self._execute_single_tool_call, tc): tc
                        for tc in tool_calls
                    }
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            tool_call_outputs.append(result)
                        except Exception as exc:
                            failed_tc = futures[future]
                            logger.error(f"Future for tool call {failed_tc.function.name} (ID: {failed_tc.id}) failed: {exc}")
                            tool_call_outputs.append({
                                "tool_call_id": failed_tc.id,
                                "role": "tool",
                                "name": failed_tc.function.name,
                                "content": json.dumps({
                                    "error": str(exc),
                                    "message": "Failed to retrieve tool result from concurrent execution",
                                }),
                            })
            else:
                tool_call_outputs.extend(
                    self._execute_single_tool_call(tc) for tc in tool_calls
                )
        try:
            msg_dict = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
            }

            if tool_calls and isinstance(tool_calls, list):
                msg_dict["tool_calls"] = self._serialize_tool_calls(tool_calls)

            messages.append(msg_dict)
            messages.extend(tool_call_outputs)
        except Exception as e:
            logger.error(f"Failed to add messages to conversation: {e}")
            raise AgentError(f"Failed to add messages to conversation: {e}") from e

        if conversation_id and self.memory:
            self._save_messages_to_memory(messages, conversation_id)

    def _should_include_artifacts_in_messages(self) -> bool:
        """Check if the memory backend supports including artifacts in messages."""
        if not self.memory:
            return False
        
        from thinagents.memory import InMemoryStore
        return isinstance(self.memory, InMemoryStore) and self.memory.store_tool_artifacts

    # Helper to prepare common streaming state
    def _prepare_stream(self, input: str, conversation_id: Optional[str]) -> List[Dict[str, Any]]:
        """Initialize artifacts and build messages for streaming runs."""
        self._tool_artifacts = {}
        return self._build_messages_with_memory(input, conversation_id)

    def _run_stream(
        self,
        input: str,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Iterator[ThinagentResponseStream[Any]]:
        """
        Streamed version of run; yields ThinagentResponseStream chunks, including interleaved tool calls/results if requested.
        """
        messages = self._prepare_stream(input, conversation_id)
        logger.info(f"Agent '{self.name}' starting streaming execution")
        
        step_count = 0
        accumulated_content = ""
        
        while step_count < self.max_steps:
            step_count += 1
            
            call_name: Optional[str] = None
            call_args: str = ""
            call_id: Optional[str] = None
            final_finish_reason: Optional[str] = None
            
            try:    
                for chunk in litellm_completion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    tools=self.tool_schemas,
                    response_format=None,
                    stream=True,
                    **self.kwargs,
                ):
                    
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        raw, opts = chunk
                        yield ThinagentResponseStream(
                            content=raw,
                            content_type="str",
                            tool_name=None,
                            tool_call_id=None,
                            response_id=None,
                            created_timestamp=None,
                            model_used=None,
                            finish_reason=None,
                            metrics=None,
                            system_fingerprint=None,
                            artifact=None,
                            stream_options=opts,
                        )
                        continue
                        
                    try:
                        sc = chunk.choices[0]  # type: ignore
                        delta = getattr(sc, "delta", None)
                        finish_reason = getattr(sc, "finish_reason", None)
                        
                        if finish_reason is not None:
                            final_finish_reason = finish_reason
                            
                    except (IndexError, AttributeError):
                        logger.warning("Invalid chunk structure in stream")
                        continue
                    
                    tool_calls = getattr(delta, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            if hasattr(tc, "function"):
                                if tc.id:
                                    call_id = tc.id
                                if tc.function.name:
                                    call_name = tc.function.name
                                if tc.function.arguments:
                                    call_args += tc.function.arguments
                                    if stream_intermediate_steps:
                                        yield ThinagentResponseStream(
                                            content=tc.function.arguments,
                                            content_type="tool_call_arg",
                                            tool_name=tc.function.name,
                                            tool_call_id=tc.id,
                                            response_id=getattr(chunk, "id", None),
                                            created_timestamp=getattr(chunk, "created", None),
                                            model_used=getattr(chunk, "model", None),
                                            finish_reason=final_finish_reason,
                                            metrics=None,
                                            system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                            artifact=None,
                                            stream_options=None,
                                        )
                    
                    fc = getattr(delta, "function_call", None)
                    if fc is not None:
                        if fc.name:
                            call_name = fc.name
                        if fc.arguments:
                            call_args += fc.arguments
                            if stream_intermediate_steps:
                                yield ThinagentResponseStream(
                                    content=fc.arguments,
                                    content_type="tool_call_arg",
                                    tool_name=fc.name if hasattr(fc, 'name') else call_name,
                                    tool_call_id=call_id,
                                    response_id=getattr(chunk, "id", None),
                                    created_timestamp=getattr(chunk, "created", None),
                                    model_used=getattr(chunk, "model", None),
                                    finish_reason=final_finish_reason,
                                    metrics=None,
                                    system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                    artifact=None,
                                    stream_options=None,
                                )
                    
                    # Check if tool/function call is complete
                    if finish_reason in ["tool_calls", "function_call"]:
                        break
                    
                    # Otherwise, stream content tokens
                    text = getattr(delta, "content", None)
                    if text:
                        accumulated_content += text  # Accumulate content
                        if self.granular_stream and len(text) > 1:
                            for ch in text:
                                yield ThinagentResponseStream(
                                    content=ch,
                                    content_type="str",
                                    tool_name=None,
                                    tool_call_id=None,
                                    response_id=getattr(chunk, "id", None),
                                    created_timestamp=getattr(chunk, "created", None),
                                    model_used=getattr(chunk, "model", None),
                                    finish_reason=final_finish_reason,
                                    metrics=None,
                                    system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                    artifact=None,
                                    stream_options=None,
                                )
                            continue
                        yield ThinagentResponseStream(
                            content=text,
                            content_type="str",
                            tool_name=None,
                            tool_call_id=None,
                            response_id=getattr(chunk, "id", None),
                            created_timestamp=getattr(chunk, "created", None),
                            model_used=getattr(chunk, "model", None),
                            finish_reason=final_finish_reason,
                            metrics=None,
                            system_fingerprint=getattr(chunk, "system_fingerprint", None),
                            artifact=None,
                            stream_options=None,
                        )
                        
                    # Check for completion without tool calls
                    if finish_reason == "stop":
                        # Save accumulated content to memory if available
                        if conversation_id and self.memory and accumulated_content:
                            final_assistant_message = {"role": "assistant", "content": accumulated_content}
                            messages.append(final_assistant_message)
                            self._save_messages_to_memory(messages, conversation_id)
                            logger.info(f"Saved final streaming response to memory for conversation '{conversation_id}'")
                            
                        logger.info(f"Agent '{self.name}' streaming completed successfully")
                        # Emit a final completion chunk to signal the end
                        yield ThinagentResponseStream(
                            content="",
                            content_type="completion",
                            tool_name=None,
                            tool_call_id=None,
                            response_id=getattr(chunk, "id", None),
                            created_timestamp=getattr(chunk, "created", None),
                            model_used=getattr(chunk, "model", None),
                            finish_reason="stop",
                            metrics=None,
                            system_fingerprint=getattr(chunk, "system_fingerprint", None),
                            artifact=None,
                            stream_options=None,
                        )
                        return
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield ThinagentResponseStream(
                    content=f"Error: {e}",
                    content_type="error",
                    tool_name=None,
                    tool_call_id=None,
                    response_id=None,
                    created_timestamp=None,
                    model_used=None,
                    finish_reason="error",
                    metrics=None,
                    system_fingerprint=None,
                    artifact=None,
                    stream_options=None,
                )
                return
            
            if call_name:
                if stream_intermediate_steps:
                    yield ThinagentResponseStream(
                        content=f"<tool_call:{call_name}>",
                        content_type="tool_call",
                        tool_name=call_name,
                        tool_call_id=call_id or f"call_{call_name}",
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=final_finish_reason,
                        metrics=None,
                        system_fingerprint=None,
                        artifact=None,
                        stream_options=None,
                    )
                
                # Parse arguments and execute tool
                try:
                    parsed_args = json.loads(call_args) if call_args else {}
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments: {e}")
                    parsed_args = {}
                    
                try:
                    tool_result = self._execute_tool(call_name, parsed_args)
                except ToolExecutionError as e:
                    logger.error(f"Tool execution failed in stream: {e}")
                    tool_result = {"error": str(e), "message": "Tool execution failed"}
                
                # determine if the tool returned artifact along with content
                tool_obj = self.tool_maps.get(call_name)
                return_type = getattr(tool_obj, "return_type", "content")
                artifact_payload = None
                if return_type == "content_and_artifact" and isinstance(tool_result, tuple) and len(tool_result) == 2:
                    content_value, artifact_payload = tool_result
                    self._tool_artifacts[call_name] = artifact_payload
                    serialised_content = self._process_tool_call_result(content_value)
                else:
                    serialised_content = self._process_tool_call_result(tool_result)

                # Optionally emit tool result with artifact (only for tool_result chunks and finish_reason==tool_calls)
                if stream_intermediate_steps:
                    yield ThinagentResponseStream(
                        content=serialised_content,
                        content_type="tool_result",
                        tool_name=call_name,
                        tool_call_id=call_id or f"call_{call_name}",
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=final_finish_reason,
                        metrics=None,
                        system_fingerprint=None,
                        artifact=self._tool_artifacts.copy() if self._tool_artifacts else None,
                        stream_options=None,
                    )
                
                # Add assistant message with tool_calls structure
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id or f"call_{call_name}",
                            "type": "function",
                            "function": {
                                "name": call_name,
                                "arguments": call_args
                            }
                        }
                    ]
                }
                messages.append(assistant_message)
                
                # Append the tool response with artifact if applicable
                tool_message: Dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call_id or f"call_{call_name}",
                    "content": serialised_content,
                }
                
                # Include artifact in tool message if memory supports it and artifacts are available
                if self._should_include_artifacts_in_messages() and artifact_payload is not None:
                    tool_message["artifact"] = artifact_payload
                
                messages.append(tool_message)
                if self.memory and conversation_id:
                    self._save_messages_to_memory(messages, conversation_id)

                
                continue
            
            break
            
        # Max steps reached in streaming
        logger.warning(f"Agent '{self.name}' reached max steps in streaming mode")
        yield ThinagentResponseStream(
            content=f"Max steps ({self.max_steps}) reached",
            content_type="error",
            tool_name=None,
            tool_call_id=None,
            response_id=None,
            created_timestamp=None,
            model_used=None,
            finish_reason="max_steps_reached",
            metrics=None,
            system_fingerprint=None,
            artifact=None,
            stream_options=None,
        )

    def __repr__(self) -> str:
        provided_tool_names = [
            getattr(t, "__name__", str(t)) for t in self._provided_tools
        ]
        
        repr_str = f"Agent(name={self.name}, model={self.model}, tools={provided_tool_names}"
        if self.sub_agents:
            sub_agent_names = [sa.name for sa in self.sub_agents]
            repr_str += f", sub_agents={sub_agent_names}"
        repr_str += ")"
        
        return repr_str

    async def _execute_tool_async(self, tool_name: str, tool_args: Dict) -> Any:
        return await asyncio.to_thread(self._execute_tool, tool_name, tool_args)

    async def _run_loop_async(self, messages: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> ThinagentResponse[_ExpectedContentType]:
        """Shared asynchronous step loop."""
        steps = 0
        json_correction_attempts = 0
        while steps < self.max_steps:
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    tools=self.tool_schemas,
                    response_format=self.response_format_model_type,
                    **self.kwargs,
                )
            except Exception as e:
                logger.error(f"LLM async completion failed: {e}")
                raise AgentError(f"LLM async completion failed: {e}") from e

            response_id = getattr(response, "id", None)
            created_timestamp = getattr(response, "created", None)
            model_used = getattr(response, "model", None)
            system_fingerprint = getattr(response, "system_fingerprint", None)
            metrics = self._extract_usage_metrics(response)

            try:
                if not hasattr(response, "choices") or not response.choices:  # type: ignore
                    logger.error("Async response has no choices")
                    raise AgentError("Invalid response structure: no choices")
                finish_reason = response.choices[0].finish_reason  # type: ignore
                message = response.choices[0].message  # type: ignore
                tool_calls = getattr(message, "tool_calls", None) or []
            except (IndexError, AttributeError) as e:
                logger.error(f"Invalid async response structure: {e}")
                raise AgentError(f"Invalid async response structure: {e}") from e

            if finish_reason == "stop" and not tool_calls:
                return self._handle_completion(
                    message, response_id, created_timestamp, model_used,
                    finish_reason, metrics, system_fingerprint, messages,
                    json_correction_attempts, conversation_id
                )

            if finish_reason == "tool_calls" or tool_calls:
                # reuse sync handler in thread to avoid blocking event loop
                await asyncio.to_thread(self._handle_tool_calls, tool_calls, message, messages, conversation_id)
                steps += 1
                continue

            steps += 1

        logger.warning(f"Agent '{self.name}' reached max steps ({self.max_steps}) in async mode")
        raise MaxStepsExceededError(f"Max steps ({self.max_steps}) reached without final answer.")

    async def _run_async(self, input: str, conversation_id: Optional[str] = None) -> ThinagentResponse[_ExpectedContentType]:
        self._tool_artifacts = {}
        messages = self._build_messages_with_memory(input, conversation_id)
        return await self._run_loop_async(messages, conversation_id)

    async def _run_stream_async(
        self,
        input: str,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[ThinagentResponseStream[Any]]:
        # artifacts initialized by helper
        logger.info(f"Agent '{self.name}' starting async streaming execution")

        messages = self._prepare_stream(input, conversation_id)
        accumulated_content = ""  # Track accumulated content for memory

        step_count = 0
        while step_count < self.max_steps:
            step_count += 1
            call_name: Optional[str] = None
            call_args: str = ""
            call_id: Optional[str] = None
            final_finish_reason: Optional[str] = None

            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    tools=self.tool_schemas,
                    response_format=None,
                    stream=True,
                    **self.kwargs,
                )

                async for chunk in response:  # type: ignore
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        raw, opts = chunk
                        yield ThinagentResponseStream(
                            content=raw,
                            content_type="str",
                            tool_name=None,
                            tool_call_id=None,
                            response_id=None,
                            created_timestamp=None,
                            model_used=None,
                            finish_reason=None,
                            metrics=None,
                            system_fingerprint=None,
                            artifact=None,
                            stream_options=opts,
                        )
                        continue

                    try:
                        sc = chunk.choices[0]  # type: ignore
                        delta = getattr(sc, "delta", None)
                        finish_reason = getattr(sc, "finish_reason", None)
                        if finish_reason is not None:
                            final_finish_reason = finish_reason
                    except (IndexError, AttributeError):
                        logger.warning("Invalid chunk structure in async stream")
                        continue

                    tool_calls = getattr(delta, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            if hasattr(tc, "function"):
                                if tc.id:
                                    call_id = tc.id
                                if tc.function.name:
                                    call_name = tc.function.name
                                if tc.function.arguments:
                                    call_args += tc.function.arguments
                                    if stream_intermediate_steps:
                                        yield ThinagentResponseStream(
                                            content=tc.function.arguments,
                                            content_type="tool_call_arg",
                                            tool_name=tc.function.name,
                                            tool_call_id=tc.id,
                                            response_id=getattr(chunk, "id", None),
                                            created_timestamp=getattr(chunk, "created", None),
                                            model_used=getattr(chunk, "model", None),
                                            finish_reason=final_finish_reason,
                                            metrics=None,
                                            system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                            artifact=None,
                                            stream_options=None,
                                        )

                    fc = getattr(delta, "function_call", None)
                    if fc is not None:
                        if fc.name:
                            call_name = fc.name
                        if fc.arguments:
                            call_args += fc.arguments
                            if stream_intermediate_steps:
                                yield ThinagentResponseStream(
                                    content=fc.arguments,
                                    content_type="tool_call_arg",
                                    tool_name=fc.name if hasattr(fc, 'name') else call_name,
                                    tool_call_id=call_id,
                                    response_id=getattr(chunk, "id", None),
                                    created_timestamp=getattr(chunk, "created", None),
                                    model_used=getattr(chunk, "model", None),
                                    finish_reason=final_finish_reason,
                                    metrics=None,
                                    system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                    artifact=None,
                                    stream_options=None,
                                )

                    if finish_reason in ["tool_calls", "function_call"]:
                        break

                    text = getattr(delta, "content", None)
                    if text:
                        accumulated_content += text  # Accumulate content
                        if self.granular_stream and len(text) > 1:
                            for ch in text:
                                yield ThinagentResponseStream(
                                    content=ch,
                                    content_type="str",
                                    tool_name=None,
                                    tool_call_id=None,
                                    response_id=getattr(chunk, "id", None),
                                    created_timestamp=getattr(chunk, "created", None),
                                    model_used=getattr(chunk, "model", None),
                                    finish_reason=final_finish_reason,
                                    metrics=None,
                                    system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                    artifact=None,
                                    stream_options=None,
                                )
                            continue
                        yield ThinagentResponseStream(
                            content=text,
                            content_type="str",
                            tool_name=None,
                            tool_call_id=None,
                            response_id=getattr(chunk, "id", None),
                            created_timestamp=getattr(chunk, "created", None),
                            model_used=getattr(chunk, "model", None),
                            finish_reason=final_finish_reason,
                            metrics=None,
                            system_fingerprint=getattr(chunk, "system_fingerprint", None),
                            artifact=None,
                            stream_options=None,
                        )

                    if finish_reason == "stop":
                        # Save accumulated content to memory if available
                        if conversation_id and self.memory and accumulated_content:
                            final_assistant_message = {"role": "assistant", "content": accumulated_content}
                            messages.append(final_assistant_message)
                            self._save_messages_to_memory(messages, conversation_id)
                            logger.info(f"Saved final streaming response to memory for conversation '{conversation_id}'")
                            
                        logger.info(f"Agent '{self.name}' async streaming completed successfully")
                        yield ThinagentResponseStream(
                            content="",
                            content_type="completion",
                            tool_name=None,
                            tool_call_id=None,
                            response_id=getattr(chunk, "id", None),
                            created_timestamp=getattr(chunk, "created", None),
                            model_used=getattr(chunk, "model", None),
                            finish_reason="stop",
                            metrics=None,
                            system_fingerprint=getattr(chunk, "system_fingerprint", None),
                            artifact=None,
                            stream_options=None,
                        )
                        return

            except Exception as e:
                logger.error(f"Async streaming error: {e}")
                yield ThinagentResponseStream(
                    content=f"Error: {e}",
                    content_type="error",
                    tool_name=None,
                    tool_call_id=None,
                    response_id=None,
                    created_timestamp=None,
                    model_used=None,
                    finish_reason="error",
                    metrics=None,
                    system_fingerprint=None,
                    artifact=None,
                    stream_options=None,
                )
                return

            if call_name:
                if stream_intermediate_steps:
                    yield ThinagentResponseStream(
                        content=f"<tool_call:{call_name}>",
                        content_type="tool_call",
                        tool_name=call_name,
                        tool_call_id=call_id or f"call_{call_name}",
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=final_finish_reason,
                        metrics=None,
                        system_fingerprint=None,
                        artifact=None,
                        stream_options=None,
                    )

                try:
                    parsed_args = json.loads(call_args) if call_args else {}
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments: {e}")
                    parsed_args = {}

                try:
                    tool_result = await self._execute_tool_async(call_name, parsed_args)
                except ToolExecutionError as e:
                    logger.error(f"Tool execution failed in async stream: {e}")
                    tool_result = {"error": str(e), "message": "Tool execution failed"}

                tool_obj = self.tool_maps.get(call_name)
                return_type = getattr(tool_obj, "return_type", "content")
                artifact_payload = None
                if return_type == "content_and_artifact" and isinstance(tool_result, tuple) and len(tool_result) == 2:
                    content_value, artifact_payload = tool_result
                    self._tool_artifacts[call_name] = artifact_payload
                    serialised_content = self._process_tool_call_result(content_value)
                else:
                    serialised_content = self._process_tool_call_result(tool_result)

                if stream_intermediate_steps:
                    yield ThinagentResponseStream(
                        content=serialised_content,
                        content_type="tool_result",
                        tool_name=call_name,
                        tool_call_id=call_id or f"call_{call_name}",
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=final_finish_reason,
                        metrics=None,
                        system_fingerprint=None,
                        artifact=self._tool_artifacts.copy() if self._tool_artifacts else None,
                        stream_options=None,
                    )

                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id or f"call_{call_name}",
                            "type": "function",
                            "function": {
                                "name": call_name,
                                "arguments": call_args
                            }
                        }
                    ]
                }
                
                messages.append(assistant_message)

                tool_message: Dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call_id or f"call_{call_name}",
                    "content": serialised_content,
                }
                
                # Include artifact in tool message if memory supports it and artifacts are available
                if self._should_include_artifacts_in_messages() and artifact_payload is not None:
                    tool_message["artifact"] = artifact_payload
                
                messages.append(tool_message)

                continue

            break

        logger.warning(f"Agent '{self.name}' reached max steps in async streaming mode")
        yield ThinagentResponseStream(
            content=f"Max steps ({self.max_steps}) reached",
            content_type="error",
            tool_name=None,
            tool_call_id=None,
            response_id=None,
            created_timestamp=None,
            model_used=None,
            finish_reason="max_steps_reached",
            metrics=None,
            system_fingerprint=None,
            artifact=None,
            stream_options=None,
        )

    @overload
    async def arun(
        self,
        input: str,
        stream: Literal[False] = False,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> ThinagentResponse[_ExpectedContentType]: ...

    @overload
    async def arun(
        self,
        input: str,
        stream: Literal[True],
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[ThinagentResponseStream[Any]]: ...

    async def arun(
        self,
        input: str,
        stream: bool = False,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Any:
        if not input or not isinstance(input, str):
            raise ValueError("Input must be a non-empty string")

        logger.info(f"Agent '{self.name}' starting async execution with input length: {len(input)}")

        if stream:
            if self.response_format_model_type:
                raise ValueError("Streaming is not supported when response_format is specified.")
            return self._run_stream_async(input, stream_intermediate_steps, conversation_id)

        return await self._run_async(input, conversation_id)

    def astream(
        self,
        input: str,
        *,
        stream_intermediate_steps: bool = False,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[ThinagentResponseStream[Any]]:

        if self.response_format_model_type:
            raise ValueError("Streaming is not supported when response_format is specified.")
        return self._run_stream_async(input, stream_intermediate_steps, conversation_id)

    def _build_messages_with_memory(self, input: str, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Build messages list including memory history if available.
        
        Args:
            input: Current user input
            conversation_id: Optional conversation ID for memory retrieval
            
        Returns:
            List of messages including system prompt, history, and current input
        """
        messages: List[Dict[str, Any]] = []
        
        # Add system prompt
        system_prompt = self._build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history from memory if available
        if self.memory and conversation_id:
            try:
                history = self.memory.get_messages(conversation_id)
                # Filter out system messages from history to avoid duplication
                # Also remove artifacts from tool messages since LLM doesn't need them
                filtered_history = []
                for msg in history:
                    if msg.get("role") == "system":
                        continue  # Skip system messages
                    
                    # Create a copy of the message for LLM
                    llm_message = msg.copy()
                    
                    # Remove artifacts from tool messages - LLM doesn't need them
                    if msg.get("role") == "tool" and "artifact" in llm_message:
                        del llm_message["artifact"]
                    
                    filtered_history.append(llm_message)
                
                messages.extend(filtered_history)
                logger.debug(f"Added {len(filtered_history)} messages from memory for conversation '{conversation_id}' (artifacts filtered out)")
            except Exception as e:
                logger.warning(f"Failed to retrieve memory for conversation '{conversation_id}': {e}")
        
        # Add current user input
        messages.append({"role": "user", "content": input})
        
        return messages

    def _save_messages_to_memory(self, messages: List[Dict], conversation_id: str) -> None:
        """
        Save new messages to memory, excluding system messages and existing history.
        
        Args:
            messages: List of all messages from the conversation
            conversation_id: Conversation ID for memory storage
        """
        if not self.memory:
            return
            
        try:
            # Get existing message count to determine which messages are new
            existing_count = self.memory.get_conversation_length(conversation_id)
            
            # Filter out system messages and get only new messages
            non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
            new_messages = non_system_messages[existing_count:]
            
            # Save new messages to memory
            for message in new_messages:
                self.memory.add_message(conversation_id, message)
                
            if new_messages:
                logger.debug(f"Saved {len(new_messages)} new messages to memory for conversation '{conversation_id}'")
        except Exception as e:
            logger.warning(f"Failed to save messages to memory for conversation '{conversation_id}': {e}")

    def clear_memory(self, conversation_id: str) -> None:
        """
        Clear memory for a specific conversation.
        
        Args:
            conversation_id: Conversation ID to clear
            
        Raises:
            ValueError: If no memory backend is configured
        """
        if not self.memory:
            raise ValueError("No memory backend configured for this agent")
        
        self.memory.clear_conversation(conversation_id)
        logger.info(f"Cleared memory for conversation '{conversation_id}'")

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a specific conversation.
        
        Args:
            conversation_id: Conversation ID to retrieve
            
        Returns:
            List of message dictionaries
            
        Raises:
            ValueError: If no memory backend is configured
        """
        if not self.memory:
            raise ValueError("No memory backend configured for this agent")
        
        return self.memory.get_messages(conversation_id)

    def list_conversations(self) -> List[ConversationInfo]:
        """
        List all conversations with detailed metadata.
        
        Returns:
            List of conversation info dictionaries with metadata
            
        Raises:
            ValueError: If no memory backend is configured
        """
        if not self.memory:
            raise ValueError("No memory backend configured for this agent")
        
        return self.memory.list_conversations()

    def list_conversation_ids(self) -> List[str]:
        """
        List all conversation IDs in memory.
        
        Returns:
            List of conversation IDs as strings
            
        Raises:
            ValueError: If no memory backend is configured
        """
        if not self.memory:
            raise ValueError("No memory backend configured for this agent")
        
        return self.memory.list_conversation_ids()

    def get_conversation_info(self, conversation_id: str) -> Optional[ConversationInfo]:
        """
        Get detailed information about a specific conversation.
        
        Args:
            conversation_id: Conversation ID to retrieve info for
            
        Returns:
            ConversationInfo dictionary or None if conversation doesn't exist
            
        Raises:
            ValueError: If no memory backend is configured
        """
        if not self.memory:
            raise ValueError("No memory backend configured for this agent")
        
        return self.memory.get_conversation_info(conversation_id)
