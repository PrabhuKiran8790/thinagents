from typing import Any, Dict, Optional, Callable

try:
    from typing import get_type_hints
except ImportError:
    from typing_extensions import get_type_hints  # type: ignore

import inspect
from thinagents.tools.tool import (
    generate_param_schema,
    is_required_parameter,
    IS_PYDANTIC_AVAILABLE,
    _BaseModel,
)
import asyncio


class LangchainTool:
    """Adapter class that wraps a Langchain tool for use with ThinAgents.

    This adapter converts Langchain tools into a format compatible with ThinAgents'
    tool interface. It preserves the tool's name, description, and functionality
    while adapting its schema.

    The original tool's name and description can be overridden if needed.

    Args:
        langchain_tool: A Langchain tool to wrap (e.g., a BaseTool instance).
        name: Optional override for the tool's name.
        description: Optional override for the tool's description.
    """

    def __init__(
        self,
        langchain_tool: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._langchain_tool = langchain_tool

        # Detect available sync and async execution methods
        sync_func = None
        if hasattr(langchain_tool, "run") and callable(getattr(langchain_tool, "run")) and not inspect.iscoroutinefunction(getattr(langchain_tool, "run")):
            sync_func = getattr(langchain_tool, "run")

        async_func = None
        if hasattr(langchain_tool, "arun") and callable(getattr(langchain_tool, "arun")) and inspect.iscoroutinefunction(getattr(langchain_tool, "arun")):
            async_func = getattr(langchain_tool, "arun")

        if sync_func is None and async_func is None:
            raise ValueError("Langchain tool must have a synchronous 'run' or an asynchronous 'arun' method.")

        self.sync_func = sync_func
        self.async_func = async_func
        
        # This is the primary function to use for schema inference and naming.
        # We prefer the sync one if available, as it's more common for schema definition.
        self.primary_func = self.sync_func or self.async_func
        if self.primary_func is None:
            # This case is defensive, the check on line 43 should prevent this.
            raise ValueError("Tool must have at least one execution method (sync or async).")
        
        # A tool is only truly async if it has no sync implementation.
        self.is_async_tool = self.sync_func is None and self.async_func is not None
        self.return_type = 'content'

        # Set name and description
        self.__name__ = name or getattr(langchain_tool, "name", self.primary_func.__name__)
        self.description = description or getattr(langchain_tool, "description", inspect.getdoc(self.primary_func) or "")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the LangChain tool synchronously."""
        if not self.sync_func:
            raise RuntimeError(f"Tool '{self.__name__}' is asynchronous and cannot be called in a synchronous agent run. Use `agent.arun()` instead.")

        tool_run_args = {k: v for k, v in kwargs.items() if k != "run_manager"}

        if len(tool_run_args) == 1:
            return self.sync_func(next(iter(tool_run_args.values())))

        return self.sync_func(tool_run_args)

    async def __acall__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the LangChain tool asynchronously."""
        if not self.async_func:
            # Fallback to running sync version in a thread if no async version exists
            if self.sync_func:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.__call__, *args, **kwargs)
            raise RuntimeError(f"Tool '{self.__name__}' does not have an asynchronous implementation.")
        
        tool_run_args = {k: v for k, v in kwargs.items() if k != "run_manager"}

        if len(tool_run_args) == 1:
            return await self.async_func(next(iter(tool_run_args.values())))
            
        return await self.async_func(tool_run_args)

    def tool_schema(self) -> Dict[str, Any]:
        """Generates the JSON schema for the LangChain tool."""
        # For tools with a Pydantic schema, use it directly.
        if IS_PYDANTIC_AVAILABLE and hasattr(self._langchain_tool, "args_schema") and self._langchain_tool.args_schema:
            if not (isinstance(self._langchain_tool.args_schema, type) and issubclass(self._langchain_tool.args_schema, _BaseModel)):
                 raise ValueError("args_schema must be a Pydantic BaseModel class")
            
            if hasattr(self._langchain_tool.args_schema, "model_json_schema"):
                 params_schema = self._langchain_tool.args_schema.model_json_schema()
            elif hasattr(self._langchain_tool.args_schema, "schema"):
                 params_schema = self._langchain_tool.args_schema.schema()
            else:
                 raise ValueError("args_schema does not have a model_json_schema or schema method.")

            # Remove Pydantic's title and OpenAI doesn't want it at the top level.
            if "title" in params_schema:
                params_schema.pop("title")
        else:
            # For simple tools, generate schema by inspecting the underlying function signature.
            # We prefer inspecting _run or func as they have named arguments.
            func_to_inspect: Optional[Callable] = None
            if hasattr(self._langchain_tool, '_run') and callable(getattr(self._langchain_tool, '_run')) and not inspect.iscoroutinefunction(self._langchain_tool._run):
                func_to_inspect = self._langchain_tool._run
            elif hasattr(self._langchain_tool, '_arun') and callable(getattr(self._langchain_tool, '_arun')) and inspect.iscoroutinefunction(self._langchain_tool._arun):
                func_to_inspect = self._langchain_tool._arun
            elif hasattr(self._langchain_tool, 'func') and callable(getattr(self._langchain_tool, 'func')):
                 func_to_inspect = self._langchain_tool.func
            
            if func_to_inspect is None:
                # Fallback for tools that only have .run() and no args_schema.
                # Assume a single string input named 'query' or 'tool_input'.
                params_schema = {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "title": "Query"}
                    },
                    "required": ["query"],
                }
            else:
                sig = inspect.signature(func_to_inspect)
                type_hints = get_type_hints(func_to_inspect, include_extras=True)
                params_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                for name, param in sig.parameters.items():
                    # Skip self, args, kwargs, and run_manager
                    if name in ("self", "args", "kwargs", "run_manager") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        continue
                    
                    annotation = type_hints.get(name, param.annotation)
                    if annotation is inspect.Parameter.empty:
                        annotation = Any
                    
                    params_schema["properties"][name] = generate_param_schema(name, param, annotation)
                    if is_required_parameter(param, annotation):
                        params_schema["required"].append(name)
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