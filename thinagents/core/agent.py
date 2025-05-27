"""
Module implementing the Agent class for orchestrating LLM interactions and tool execution.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Iterator, TypeVar, Generic, cast, overload, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import litellm
from litellm import completion as litellm_completion
from pydantic import BaseModel, ValidationError # type: ignore
from thinagents.core.tool import ThinAgentsTool, tool as tool_decorator
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
        parallel_tool_calls: bool = False,
        concurrent_tool_execution: bool = True,
        response_format: Optional[Type[_ExpectedContentType]] = None,
        enable_schema_validation: bool = True,
        description: Optional[str] = None,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
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

        self.parallel_tool_calls = parallel_tool_calls
        self.concurrent_tool_execution = concurrent_tool_execution
        self.kwargs = kwargs

        self._provided_tools = tools or []

        # Whether to emit per-character ThinagentResponseStream chunks when streaming
        self.granular_stream = True

        # Initialize tools and sub-agents
        self._initialize_tools()

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

    @overload
    def run(
        self,
        input: str,
        stream: Literal[False] = False,
        stream_intermediate_steps: bool = False,
    ) -> ThinagentResponse[_ExpectedContentType]:
        ...
    @overload
    def run(
        self,
        input: str,
        stream: Literal[True],
        stream_intermediate_steps: bool = False,
    ) -> Iterator[ThinagentResponseStream[Any]]:
        ...
    def run(
        self,
        input: str,
        stream: bool = False,
        stream_intermediate_steps: bool = False,
    ) -> Any:
        """
        Run the agent with the given input and manage interactions with the language model and tools.

        Args:
            input: The user's input message to the agent.
            stream: If True, returns a stream of responses instead of a single response.
            stream_intermediate_steps: If True and stream=True, also stream intermediate tool calls and results.

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
            return self._run_stream(input, stream_intermediate_steps)

        try:
            return self._run_sync(input)
        except Exception as e:
            logger.error(f"Agent '{self.name}' execution failed: {e}")
            if isinstance(e, (AgentError, MaxStepsExceededError)):
                raise
            raise AgentError(f"Agent execution failed: {e}") from e

    def _run_sync(self, input: str) -> ThinagentResponse[_ExpectedContentType]:
        """Synchronous execution of the agent."""
        # initialize storage for tool artifacts
        self._tool_artifacts: dict[str, Any] = {}
        steps = 0
        json_correction_attempts = 0
        messages: List[Dict] = []

        system_prompt = self._build_system_prompt()
        messages.extend(
            (
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input},
            )
        )
        while steps < self.max_steps:
            try:
                response = litellm_completion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    tools=self.tool_schemas,
                    parallel_tool_calls=self.parallel_tool_calls,
                    response_format=self.response_format_model_type,
                    **self.kwargs,
                )
            except Exception as e:
                logger.error(f"LLM completion failed: {e}")
                raise AgentError(f"LLM completion failed: {e}") from e

            # Extract response metadata
            response_id = getattr(response, "id", None)
            created_timestamp = getattr(response, "created", None)
            model_used = getattr(response, "model", None)
            system_fingerprint = getattr(response, "system_fingerprint", None)
            metrics = self._extract_usage_metrics(response)

            # Process response
            try:
                if not hasattr(response, 'choices') or not response.choices: # type: ignore
                    logger.error("Response has no choices")
                    raise AgentError("Invalid response structure: no choices")

                finish_reason = response.choices[0].finish_reason # type: ignore
                message = response.choices[0].message # type: ignore
                tool_calls = getattr(message, "tool_calls", None) or []
            except (IndexError, AttributeError) as e:
                logger.error(f"Invalid response structure: {e}")
                raise AgentError(f"Invalid response structure: {e}") from e

            # Handle completion without tool calls
            if finish_reason == "stop" and not tool_calls:
                return self._handle_completion(
                    message, response_id, created_timestamp, model_used, 
                    finish_reason, metrics, system_fingerprint, messages, 
                    json_correction_attempts
                )

            # Handle tool calls
            if finish_reason == "tool_calls" or tool_calls:
                self._handle_tool_calls(tool_calls, message, messages)
                steps += 1
                continue

            steps += 1

        # Max steps reached
        logger.warning(f"Agent '{self.name}' reached max steps ({self.max_steps})")
        raise MaxStepsExceededError(f"Max steps ({self.max_steps}) reached without final answer.")

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
        json_correction_attempts: int
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
                    return self._run_sync(messages[-1]["content"])
                # Max attempts reached, return error
                final_content = cast(_ExpectedContentType, f"JSON validation failed after {MAX_JSON_CORRECTION_ATTEMPTS} attempts: {e}")
                content_type_to_return = "str"
        else:
            final_content = cast(_ExpectedContentType, raw_content_from_llm)
            content_type_to_return = "str"

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

    def _handle_tool_calls(self, tool_calls: List[Any], message: Any, messages: List[Dict]) -> None:
        """Handle tool calls execution."""
        tool_call_outputs = []
        
        if tool_calls:
            def _process_individual_tool_call(tc: Any) -> Dict[str, Any]:
                tool_call_name = tc.function.name
                tool_call_id = tc.id
                # check if tool returns artifact
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
                    # unpack content and artifact if provided
                    if return_type == "content_and_artifact" and isinstance(raw_result, tuple) and len(raw_result) == 2:
                        content_value, artifact = raw_result
                        self._tool_artifacts[tool_call_name] = artifact
                    else:
                        content_value = raw_result
                    content_for_llm = self._process_tool_call_result(content_value)
                    
                    return {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": tool_call_name,
                        "content": content_for_llm,
                    }
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

            # Execute tool calls (concurrent or sequential)
            if self.concurrent_tool_execution and len(tool_calls) > 1:
                with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
                    futures = {
                        executor.submit(_process_individual_tool_call, tc): tc
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
                for tc in tool_calls:
                    tool_call_outputs.append(_process_individual_tool_call(tc))

        # Add assistant message and tool outputs to conversation
        try:
            if hasattr(message, "__dict__"):
                msg_dict = dict(message.__dict__)
                msg_dict = {k: v for k, v in msg_dict.items() if not k.startswith("_")}
                if "role" not in msg_dict and hasattr(message, "role"):
                    msg_dict["role"] = message.role
                if "content" not in msg_dict and hasattr(message, "content"):
                    msg_dict["content"] = message.content
            else:
                msg_dict = {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", ""),
                }

            messages.append(msg_dict)
            messages.extend(tool_call_outputs)
        except Exception as e:
            logger.error(f"Failed to add messages to conversation: {e}")
            raise AgentError(f"Failed to add messages to conversation: {e}") from e

    def _run_stream(
        self,
        input: str,
        stream_intermediate_steps: bool = False,
    ) -> Iterator[ThinagentResponseStream[Any]]:
        """
        Streamed version of run; yields ThinagentResponseStream chunks, including interleaved tool calls/results if requested.
        """
        # initialise storage for tool artifacts for this run
        self._tool_artifacts = {}
        logger.info(f"Agent '{self.name}' starting streaming execution")
        
        # Build initial messages
        system_prompt = self._build_system_prompt()
        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input},
        ]
        
        step_count = 0
        while step_count < self.max_steps:
            step_count += 1
            
            # Accumulate function call args across deltas
            call_name: Optional[str] = None
            call_args: str = ""
            call_id: Optional[str] = None
            final_finish_reason: Optional[str] = None
            
            try:
                # Stream chat until a function call is completed or done
                for chunk in litellm_completion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    tools=self.tool_schemas,
                    parallel_tool_calls=self.parallel_tool_calls,
                    response_format=None,
                    stream=True,
                    **self.kwargs,
                ):
                    
                    # Raw tuple from custom streams
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
                        
                    # Standard streaming choice
                    try:
                        sc = chunk.choices[0]
                        delta = getattr(sc, "delta", None)
                        finish_reason = getattr(sc, "finish_reason", None)
                        
                        # Track the final finish reason when it appears
                        if finish_reason is not None:
                            final_finish_reason = finish_reason
                            
                    except (IndexError, AttributeError):
                        logger.warning("Invalid chunk structure in stream")
                        continue
                    
                    # Handle tool_calls (modern format)
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
                    
                    # Handle function_call (legacy format)
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
            
            # If a function call was made, execute and loop again
            if call_name:
                # Optionally emit tool call event
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
                assistant_message = {
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
                
                # Append the tool response
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id or f"call_{call_name}",
                    "content": serialised_content,
                }
                messages.append(tool_message)
                
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
