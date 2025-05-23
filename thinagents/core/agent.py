"""
Module implementing the Agent class for orchestrating LLM interactions and tool execution.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Iterator, TypeVar, Generic, cast, overload, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from litellm import completion as litellm_completion
from pydantic import BaseModel # type: ignore
from thinagents.core.tool import ThinAgentsTool, tool as tool_decorator
from thinagents.utils.prompts import PromptConfig
from thinagents.core.response_models import (
    ThinagentResponse as GenericThinagentResponse,
    ThinagentResponseStream,
    UsageMetrics,
    CompletionTokensDetails,
    PromptTokensDetails,
)

_ExpectedContentType = TypeVar('_ExpectedContentType', bound=BaseModel)


def generate_tool_schemas(
    tools: Union[List[ThinAgentsTool], List[Callable]],
) -> Tuple[List[Dict], Dict[str, ThinAgentsTool]]:
    """
    Generate JSON schemas for provided tools and return tool schemas list and tool maps.

    Args:
        tools: A list of ThinAgentsTool instances or callables decorated with @tool.

    Returns:
        Tuple[List[Dict], Dict[str, ThinAgentsTool]]: A list of tool schema dictionaries and a mapping from tool names to ThinAgentsTool instances.
    """
    tool_schemas = []
    tool_maps: Dict[str, ThinAgentsTool] = {}

    for tool in tools:
        if isinstance(tool, ThinAgentsTool):
            tool_schemas.append(tool.tool_schema())
            tool_maps[tool.__name__] = tool
        else:
            _tool = tool_decorator(tool)
            tool_schemas.append(_tool.tool_schema())
            tool_maps[_tool.__name__] = _tool
    return tool_schemas, tool_maps


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
        max_steps: int = 15,
        parallel_tool_calls: bool = False,
        concurrent_tool_execution: bool = True,
        response_format: Optional[Type[_ExpectedContentType]] = None,
        enable_schema_validation: bool = True,
        description: Optional[str] = None,
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
            granular_stream: If True and stream=True, also stream character-level chunks for delta.content.
            **kwargs: Additional keyword arguments that will be passed directly to the `litellm.completion` function.
        """

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

        def _make_sub_agent_tool(sa: "Agent") -> ThinAgentsTool:
            """Create a ThinAgents tool that delegates calls to a sub-agent."""
            safe_name = sa.name.lower().strip().replace(" ", "_")

            def _delegate_to_sub_agent(input: str):
                """Delegate input to the sub-agent."""
                return sa.run(input)

            _delegate_to_sub_agent.__name__ = f"subagent_{safe_name}"

            _delegate_to_sub_agent.__doc__ = sa.description or (
                f"Forward the input to the '{sa.name}' sub-agent and return its response."
            )

            return tool_decorator(_delegate_to_sub_agent)

        sub_agent_tools: List[ThinAgentsTool] = [_make_sub_agent_tool(sa) for sa in self.sub_agents]

        combined_tools = (tools or []) + sub_agent_tools
        self.tool_schemas, self.tool_maps = generate_tool_schemas(combined_tools)

    def _execute_tool(self, tool_name: str, tool_args: dict):
        """Executes a tool by name with the provided arguments."""
        tool = self.tool_maps.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return tool(**tool_args)

    @overload
    def run(
        self,
        input: str,
        stream: Literal[False] = False,
        stream_intermediate_steps: bool = False,
    ) -> GenericThinagentResponse[_ExpectedContentType]:
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
            GenericThinagentResponse[_ExpectedContentType] when stream=False, or Iterator[ThinagentResponseStream] when stream=True.
        """
        # Handle streaming response
        if stream:
            if self.response_format_model_type:
                raise ValueError("Streaming is not supported when response_format is specified.")
            return self._run_stream(input, stream_intermediate_steps)

        steps = 0
        messages: List[Dict] = []

        if isinstance(self.prompt, PromptConfig):
            system_prompt = self.prompt.add_instruction(
                f"Your name is {self.name}"
            ).build()
        else:
            if self.prompt is None:
                base_prompt = (
                    f"""
                    You are a helpful assistant. Answer the user's question to the best of your ability.
                    You are given a name, your name is {self.name}.
                    """
                )
            else:
                base_prompt = self.prompt

            if self.instructions:
                base_prompt = f"{base_prompt}\n" + "\n".join(self.instructions)

            system_prompt = base_prompt

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input})

        while steps < self.max_steps:
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

            response_id = getattr(response, "id", None)
            created_timestamp = getattr(response, "created", None)
            model_used = getattr(response, "model", None)
            system_fingerprint = getattr(response, "system_fingerprint", None)
            raw_usage = getattr(response, "usage", None)
            metrics = None

            if raw_usage:
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
                
                metrics = UsageMetrics(
                    completion_tokens=getattr(raw_usage, "completion_tokens", None),
                    prompt_tokens=getattr(raw_usage, "prompt_tokens", None),
                    total_tokens=getattr(raw_usage, "total_tokens", None),
                    completion_tokens_details=ct_details,
                    prompt_tokens_details=pt_details,
                )

            finish_reason = response.choices[0].finish_reason  # type: ignore
            message = response.choices[0].message  # type: ignore
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None:
                tool_calls = []

            if finish_reason == "stop" and not tool_calls:
                raw_content_from_llm = message.content
                final_content: _ExpectedContentType
                content_type_to_return: str

                if self.response_format_model_type:
                    try:
                        parsed_model = self.response_format_model_type.model_validate_json(
                            raw_content_from_llm  # type: ignore
                        )
                        final_content = cast(_ExpectedContentType, parsed_model)
                        content_type_to_return = self.response_format_model_type.__name__
                    except Exception as e:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"The JSON is invalid: {e}. Please fix the JSON and return it. Returned JSON: {raw_content_from_llm}, Expected JSON: {self.response_format_model_type.model_json_schema()}",
                            }
                        )
                        correction_response = litellm_completion(
                            model=self.model,
                            messages=messages,
                            api_key=self.api_key,
                            api_base=self.api_base,
                            api_version=self.api_version,
                            **self.kwargs,
                        )
                        message = correction_response.choices[0].message # type: ignore
                        messages.append({"role": message.role, "content": message.content})
                        steps += 1 
                        continue 
                else: # Expecting a string
                    final_content = cast(_ExpectedContentType, raw_content_from_llm)
                    content_type_to_return = "str"
                
                return GenericThinagentResponse(
                    content=final_content,
                    content_type=content_type_to_return,
                    response_id=response_id,
                    created_timestamp=created_timestamp,
                    model_used=model_used,
                    finish_reason=finish_reason,
                    metrics=metrics,
                    system_fingerprint=system_fingerprint,
                    extra_data=None
                )

            if finish_reason == "tool_calls" or tool_calls:
                tool_call_outputs = []
                if tool_calls:
                    def _process_individual_tool_call(tc):
                        tool_call_name = tc.function.name
                        tool_call_id = tc.id
                        try:
                            tool_call_args = json.loads(tc.function.arguments)
                        except Exception as e:
                            print(
                                f"Error parsing tool arguments for {tool_call_name} (ID: {tool_call_id}): {e}"
                            )
                            return {
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "name": tool_call_name,
                                "content": json.dumps(
                                    {
                                        "error": str(e),
                                        "message": "Failed to parse arguments",
                                    }
                                ),
                            }
                        try:
                            tool_call_result = self._execute_tool(
                                tool_call_name, tool_call_args
                            )

                            content_for_llm: str
                            if isinstance(tool_call_result, GenericThinagentResponse):
                                # Result from a sub-agent
                                sub_agent_content_data = tool_call_result.content
                                if isinstance(sub_agent_content_data, BaseModel): 
                                    content_for_llm = sub_agent_content_data.model_dump_json()
                                elif isinstance(sub_agent_content_data, str):
                                    content_for_llm = sub_agent_content_data
                                else: # dict, list, int, float etc. from sub_agent_content_data
                                    content_for_llm = json.dumps(sub_agent_content_data)
                            elif isinstance(tool_call_result, BaseModel):
                                content_for_llm = tool_call_result.model_dump_json()
                            elif isinstance(tool_call_result, str):
                                content_for_llm = tool_call_result
                            else: # dict, list, int, float etc. from a regular tool
                                content_for_llm = json.dumps(tool_call_result)

                            return {
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "name": tool_call_name,
                                "content": content_for_llm,
                            }
                        except Exception as e:
                            print(
                                f"Error executing tool {tool_call_name} (ID: {tool_call_id}): {e}"
                            )
                            return {
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "name": tool_call_name,
                                "content": json.dumps(
                                    {
                                        "error": str(e),
                                        "message": "Tool execution failed",
                                    }
                                ),
                            }

                    if self.concurrent_tool_execution and len(tool_calls) > 1:
                        with ThreadPoolExecutor(
                            max_workers=len(tool_calls)
                        ) as executor:
                            futures = {
                                executor.submit(_process_individual_tool_call, tc): tc
                                for tc in tool_calls
                            }
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                    tool_call_outputs.append(result)
                                except Exception as exc:
                                    failed_tc = futures[future]
                                    print(
                                        f"Future for tool call {failed_tc.function.name} (ID: {failed_tc.id}) generated an exception: {exc}"
                                    )
                                    tool_call_outputs.append(
                                        {
                                            "tool_call_id": failed_tc.id,
                                            "role": "tool",
                                            "name": failed_tc.function.name,
                                            "content": json.dumps(
                                                {
                                                    "error": str(exc),
                                                    "message": "Failed to retrieve tool result from concurrent execution",
                                                }
                                            ),
                                        }
                                    )
                    else:
                        for tc in tool_calls:
                            tool_call_outputs.append(_process_individual_tool_call(tc))

                if hasattr(message, "__dict__"):
                    msg_dict = dict(message.__dict__)
                    msg_dict = {
                        k: v for k, v in msg_dict.items() if not k.startswith("_")
                    }
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
                steps += 1
                continue

            steps += 1

        final_content_on_max_steps = cast(_ExpectedContentType, "Max steps reached without final answer.")
        return GenericThinagentResponse(
            content=final_content_on_max_steps,
            content_type="str",
            response_id=response_id, 
            created_timestamp=created_timestamp, 
            model_used=model_used, 
            finish_reason="max_steps_reached",
            metrics=metrics, 
            system_fingerprint=system_fingerprint, 
            extra_data=None
        )

    def _run_stream(
        self,
        input: str,
        stream_intermediate_steps: bool = False,
    ) -> Iterator[ThinagentResponseStream[Any]]:
        """
        Streamed version of run; yields ThinagentResponseStream chunks, including interleaved tool calls/results if requested.
        """
        # Build initial messages
        if isinstance(self.prompt, PromptConfig):
            system_prompt = self.prompt.add_instruction(f"Your name is {self.name}").build()
        else:
            base = self.prompt or f"You are a helpful assistant. You are {self.name}."
            if self.instructions:
                base = f"{base}\n" + "\n".join(self.instructions)
            system_prompt = base
        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input},
        ]
        # Repeat until no more function calls
        step_count = 0
        while True:
            step_count += 1
            
            # Accumulate function call args across deltas
            call_name: Optional[str] = None
            call_args: str = ""
            call_id: Optional[str] = None
            
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
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=None,
                        metrics=None,
                        system_fingerprint=None,
                        extra_data=None,
                        stream_options=opts,
                    )
                    continue
                    
                # Standard streaming choice
                sc = chunk.choices[0]
                delta = getattr(sc, "delta", None)
                finish_reason = getattr(sc, "finish_reason", None)
                
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
                                        response_id=None,
                                        created_timestamp=None,
                                        model_used=None,
                                        finish_reason=None,
                                        metrics=None,
                                        system_fingerprint=None,
                                        extra_data=None,
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
                                response_id=None,
                                created_timestamp=None,
                                model_used=None,
                                finish_reason=None,
                                metrics=None,
                                system_fingerprint=None,
                                extra_data=None,
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
                                response_id=getattr(chunk, "id", None),
                                created_timestamp=getattr(chunk, "created", None),
                                model_used=getattr(chunk, "model", None),
                                finish_reason=finish_reason,
                                metrics=None,
                                system_fingerprint=getattr(chunk, "system_fingerprint", None),
                                extra_data=None,
                                stream_options=None,
                            )
                        continue
                    yield ThinagentResponseStream(
                        content=text,
                        content_type="str",
                        response_id=getattr(chunk, "id", None),
                        created_timestamp=getattr(chunk, "created", None),
                        model_used=getattr(chunk, "model", None),
                        finish_reason=finish_reason,
                        metrics=None,
                        system_fingerprint=getattr(chunk, "system_fingerprint", None),
                        extra_data=None,
                        stream_options=None,
                    )
                    
                # Check for completion without tool calls
                if finish_reason == "stop":
                    return
            
            # If a function call was made, execute and loop again
            if call_name:
                
                # Optionally emit tool call event
                if stream_intermediate_steps:
                    yield ThinagentResponseStream(
                        content=f"<tool_call:{call_name}>",
                        content_type="tool_call",
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=None,
                        metrics=None,
                        system_fingerprint=None,
                        extra_data=None,
                        stream_options=None,
                    )
                
                # Parse arguments and execute tool
                try:
                    parsed_args = json.loads(call_args) if call_args else {}
                except Exception as e:
                    parsed_args = {}
                    
                tool_result = self._execute_tool(call_name, parsed_args)
                
                # Optionally emit tool result
                if stream_intermediate_steps:
                    yield ThinagentResponseStream(
                        content=json.dumps(tool_result),
                        content_type="tool_result",
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=None,
                        metrics=None,
                        system_fingerprint=None,
                        extra_data=None,
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
                    "content": json.dumps(tool_result)
                }
                messages.append(tool_message)
                
                continue
            
            break

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
