"""
Implementation of the Agent class
"""

import json
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from litellm import completion as litellm_completion
from thinagents.core.tool import ThinAgentsTool, tool as tool_decorator
from thinagents.utils.prompts import PromptConfig
from pydantic import BaseModel


def generate_tool_schemas(
    tools: Union[List[ThinAgentsTool], List[Callable]],
) -> Tuple[List[Dict], Dict[str, ThinAgentsTool]]:
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


class Agent:
    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        tools: Optional[Union[List[ThinAgentsTool], List[Callable]]] = None,
        prompt: Optional[Union[str, PromptConfig]] = None,
        max_steps: int = 15,
        parallel_tool_calls: bool = False,
        concurrent_tool_execution: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        enable_schema_validation: bool = True,
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
            prompt: The system prompt to guide the agent's behavior.
                This can be a simple string or a `PromptConfig` object for more complex prompt engineering.
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
            **kwargs: Additional keyword arguments that will be passed directly to the `litellm.completion` function.
        """

        self.name = name
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.max_steps = max_steps
        self.prompt = prompt
        self.tool_schemas, self.tool_maps = generate_tool_schemas(tools or [])
        self.parallel_tool_calls = parallel_tool_calls
        self.concurrent_tool_execution = concurrent_tool_execution
        self.kwargs = kwargs

        self.response_format = response_format
        self.enable_schema_validation = enable_schema_validation
        if self.response_format:
            litellm.enable_json_schema_validation = self.enable_schema_validation

    def _execute_tool(self, tool_name: str, tool_args: dict):
        """Executes a tool by name with the provided arguments."""
        tool = self.tool_maps.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return tool(**tool_args)

    def run(self, input: str) -> str:
        steps = 0
        messages: List[Dict] = []

        if self.prompt is None:
            messages.append(
                {
                    "role": "system",
                    "content": f"""
                    You are a helpful assistant. Answer the user's question to the best of your ability.
                    You are given a name, your name is {self.name}.
                    """,
                }
            )
        else:
            system_prompt = (
                self.prompt.add_instruction(f"Your name is {self.name}").build()
                if isinstance(self.prompt, PromptConfig)
                else self.prompt
            )
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
                tool_choice="auto",
                parallel_tool_calls=self.parallel_tool_calls,
                response_format=self.response_format,
                **self.kwargs,
            )

            finish_reason = response.choices[0].finish_reason  # type: ignore
            message = response.choices[0].message  # type: ignore
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None:
                tool_calls = []

            if finish_reason == "stop" and not tool_calls:
                if not self.response_format:
                    return message.content  # type: ignore
                else:
                    try:
                        self.response_format.model_validate_json(
                            message.content  # type: ignore
                        )
                        return message.content  # type: ignore
                    except Exception as e:
                        # ask LLM to fix the error and return the fixed JSON
                        messages.append(
                            {
                                "role": "user",
                                "content": f"The JSON is invalid: {e}. Please fix the JSON and return it. Returned JSON: {message.content}, Expected JSON: {self.response_format.model_json_schema()}",
                            }
                        )
                        response = litellm_completion(
                            model=self.model,
                            messages=messages,
                            **self.kwargs,
                        )
                        # do not return directly, continue the loop and if it was fixed, our above code will return the fixed JSON
                        continue

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
                            return {
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "name": tool_call_name,
                                "content": (
                                    tool_call_result
                                    if isinstance(tool_call_result, str)
                                    else json.dumps(tool_call_result)
                                ),
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
                        # Concurrent execution for tool calls
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

        return "Max steps reached without final answer."

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, model={self.model}, tools={[t for t in self.tool_maps.keys()]})"
