"""
CrewAI tool adapter for ThinAgents.

This module provides an adapter to integrate CrewAI tools with ThinAgents,
allowing CrewAI BaseTool instances to be used seamlessly within ThinAgents agents.

Example usage:
    ```python
    from crewai.tools import BaseTool
    from thinagents.tools.crewai_tool import CrewaiTool
    from thinagents import Agent

    # Define a CrewAI tool
    class WeatherTool(BaseTool):
        name: str = "weather_tool"
        description: str = "Get weather information for a city"
        
        def _run(self, city: str) -> str:
            # Your weather API logic here
            return f"The weather in {city} is sunny."

    # Create the CrewAI tool instance
    weather_tool = WeatherTool()

    # Wrap it with the ThinAgents adapter
    wrapped_tool = CrewaiTool(weather_tool)

    # Use it in a ThinAgents agent
    agent = Agent(
        name="Weather Agent",
        model="openai/gpt-4o-mini",
        tools=[wrapped_tool]
    )

    # The agent can now use the CrewAI tool
    response = await agent.arun("What's the weather like in New York?")
    print(response.content)
    ```
"""

import asyncio
import inspect
import json
import re
from typing import Any, Dict, Optional, Callable

try:
    from typing import get_type_hints
except ImportError:
    from typing_extensions import get_type_hints  # type: ignore

from thinagents.tools.tool import (
    generate_param_schema,
    is_required_parameter,
    IS_PYDANTIC_AVAILABLE,
    _BaseModel,
    sanitize_function_name,
    FunctionNameSanitizationError,
)

# Check if CrewAI is available
try:
    # CrewAI typically imports as crewai
    import crewai  # type: ignore
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


class CrewaiIntegrationError(Exception):
    """Custom exception for CrewAI integration errors."""
    pass



class CrewaiTool:
    """
    Adapter class that wraps a CrewAI tool for use with ThinAgents.

    This adapter converts CrewAI tools into a format compatible with ThinAgents'
    tool interface. It preserves the tool's name, description, and functionality
    while adapting its schema and execution methods.

    The original tool's name and description can be overridden if needed.

    Args:
        crewai_tool: A CrewAI tool instance to wrap (e.g., a BaseTool instance).
        name: Optional override for the tool's name.
        description: Optional override for the tool's description.
    """

    def __init__(
        self,
        crewai_tool: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        if not CREWAI_AVAILABLE:
            raise CrewaiIntegrationError(
                "The 'crewai' library is required to use CrewAI tools. "
                "Please install it with 'pip install crewai' or 'pip install crewai[tools]'."
            )

        self._crewai_tool = crewai_tool

        # Check if the tool has the required interface
        if not hasattr(crewai_tool, '_run'):
            raise CrewaiIntegrationError(
                "CrewAI tool must have a '_run' method. "
                "Ensure the tool is properly implemented as a CrewAI BaseTool."
            )

        # Detect available sync and async execution methods
        sync_func = None
        if hasattr(crewai_tool, '_run') and callable(getattr(crewai_tool, '_run')):
            sync_func = getattr(crewai_tool, '_run')

        async_func = None
        if hasattr(crewai_tool, '_arun') and callable(getattr(crewai_tool, '_arun')):
            async_func = getattr(crewai_tool, '_arun')

        # CrewAI tools primarily use _run, but some might have _arun
        if sync_func is None and async_func is None:
            raise CrewaiIntegrationError(
                "CrewAI tool must have either a '_run' or '_arun' method."
            )

        self.sync_func = sync_func
        self.async_func = async_func
        
        # Primary function for schema inference
        self.primary_func = self.sync_func or self.async_func
        
        # A tool is only truly async if it has no sync implementation
        self.is_async_tool = self.sync_func is None and self.async_func is not None
        self.return_type = 'content'

        # Set name and description from the CrewAI tool or use overrides
        raw_name = name or getattr(crewai_tool, 'name', 'crewai_tool')
        try:
            self.__name__ = sanitize_function_name(raw_name)
        except FunctionNameSanitizationError as e:
            raise CrewaiIntegrationError(f"Failed to create valid tool name for CrewAI tool: {e}") from e
        self.description = description or getattr(crewai_tool, 'description', '')

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the CrewAI tool synchronously."""
        if not self.sync_func:
            raise RuntimeError(
                f"CrewAI tool '{self.__name__}' is asynchronous and cannot be called "
                "in a synchronous agent run. Use `agent.arun()` instead."
            )

        try:
            # CrewAI tools typically expect a single string input or keyword arguments
            # Handle different calling patterns
            if len(args) == 1 and len(kwargs) == 0:
                # Single positional argument
                return self.sync_func(args[0])
            elif len(args) == 0 and len(kwargs) > 0:
                # Keyword arguments only
                return self.sync_func(**kwargs)
            elif len(args) == 0 and len(kwargs) == 0:
                # No arguments
                return self.sync_func()
            else:
                # Mixed arguments - convert to kwargs if possible
                if len(args) == 1:
                    return self.sync_func(args[0], **kwargs)
                else:
                    return self.sync_func(*args, **kwargs)
        except Exception as e:
            raise CrewaiIntegrationError(f"CrewAI tool execution failed: {e}") from e

    async def __acall__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the CrewAI tool asynchronously."""
        if self.async_func:
            # Use native async method if available
            try:
                if len(args) == 1 and len(kwargs) == 0:
                    return await self.async_func(args[0])
                elif len(args) == 0 and len(kwargs) > 0:
                    return await self.async_func(**kwargs)
                elif len(args) == 0 and len(kwargs) == 0:
                    return await self.async_func()
                else:
                    if len(args) == 1:
                        return await self.async_func(args[0], **kwargs)
                    else:
                        return await self.async_func(*args, **kwargs)
            except Exception as e:
                raise CrewaiIntegrationError(f"CrewAI async tool execution failed: {e}") from e
        elif self.sync_func:
            # Fallback to running sync version in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.__call__, *args, **kwargs)
        else:
            raise RuntimeError(
                f"CrewAI tool '{self.__name__}' does not have an execution method."
            )

    def tool_schema(self) -> Dict[str, Any]:
        """Generate the JSON schema for the CrewAI tool."""
        # Check if the tool has a Pydantic schema
        if IS_PYDANTIC_AVAILABLE and hasattr(self._crewai_tool, 'args_schema') and self._crewai_tool.args_schema:
            if not (isinstance(self._crewai_tool.args_schema, type) and issubclass(self._crewai_tool.args_schema, _BaseModel)):
                raise ValueError("args_schema must be a Pydantic BaseModel class")
            
            if hasattr(self._crewai_tool.args_schema, "model_json_schema"):
                params_schema = self._crewai_tool.args_schema.model_json_schema()
            elif hasattr(self._crewai_tool.args_schema, "schema"):
                params_schema = self._crewai_tool.args_schema.schema()
            else:
                raise ValueError("args_schema does not have a model_json_schema or schema method.")

            # Remove Pydantic's title as OpenAI doesn't want it at the top level
            if "title" in params_schema:
                params_schema.pop("title")
        else:
            # Generate schema by inspecting the _run method signature
            func_to_inspect: Optional[Callable] = None
            
            # Try to find the best function to inspect
            if hasattr(self._crewai_tool, '_run') and callable(getattr(self._crewai_tool, '_run')):
                func_to_inspect = self._crewai_tool._run
            elif hasattr(self._crewai_tool, '_arun') and callable(getattr(self._crewai_tool, '_arun')):
                func_to_inspect = self._crewai_tool._arun
            
            if func_to_inspect is None:
                # Fallback for tools without clear method signatures
                # CrewAI tools typically accept a single string input
                params_schema = {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "title": "Input"}
                    },
                    "required": ["input"],
                }
            else:
                # Inspect the function signature
                sig = inspect.signature(func_to_inspect)
                type_hints = get_type_hints(func_to_inspect, include_extras=True)
                params_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                
                for name, param in sig.parameters.items():
                    # Skip self and other internal parameters
                    if name in ("self", "args", "kwargs") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        continue
                    
                    annotation = type_hints.get(name, param.annotation)
                    if annotation is inspect.Parameter.empty:
                        annotation = Any
                    
                    params_schema["properties"][name] = generate_param_schema(name, param, annotation)
                    if is_required_parameter(param, annotation):
                        params_schema["required"].append(name)
                
                # If no parameters found, assume single string input (common for CrewAI tools)
                if not params_schema["properties"]:
                    params_schema = {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "title": "Input"}
                        },
                        "required": ["input"],
                    }
                
                params_schema["required"] = sorted(list(set(params_schema["required"])))

        # Construct the final OpenAI-compatible schema
        final_schema = {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.description,
                "parameters": params_schema,
            },
        }
        
        return {"tool_schema": final_schema, "return_type": self.return_type} 