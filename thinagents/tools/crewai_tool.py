from typing import Any, Dict, Optional
import asyncio
import inspect
import re

from thinagents.tools.tool import (
    IS_PYDANTIC_AVAILABLE,
    _BaseModel,
)

def sanitize_function_name(name: str) -> str:
    """
    Sanitize a function name to comply with LLM function calling requirements.
    
    Requirements:
    - Must start with a letter or underscore
    - Can contain letters, numbers, underscores, dots, and dashes
    - Maximum length of 64 characters
    
    Args:
        name: Original function name
        
    Returns:
        Sanitized function name that's compatible with LLM function calling
    """
    if not name:
        return "unnamed_tool"
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    
    # Ensure it starts with letter or underscore
    if not re.match(r'^[a-zA-Z_]', sanitized):
        sanitized = f"tool_{sanitized}"
    
    # Truncate to 64 characters
    if len(sanitized) > 64:
        sanitized = sanitized[:64]
    
    return sanitized

class CrewAITool:
    """Adapter class that wraps a CrewAI tool for use with ThinAgents."""

    def __init__(
        self,
        crewai_tool: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._crewai_tool = crewai_tool

        if not hasattr(crewai_tool, "_run") or not callable(getattr(crewai_tool, "_run")):
            raise ValueError("CrewAI tool must have a callable '_run' method.")
        
        if not hasattr(crewai_tool, "run") or not callable(getattr(crewai_tool, "run")):
            raise ValueError("CrewAI tool must have a callable 'run' method.")

        self.is_async_tool = inspect.iscoroutinefunction(self._crewai_tool._run)
        
        self.return_type = 'content'

        raw_name = name or getattr(crewai_tool, "name")
        self.__name__ = sanitize_function_name(raw_name)
        self.description = description or getattr(crewai_tool, "description", "")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the CrewAI tool synchronously."""
        if self.is_async_tool:
            raise RuntimeError(f"Tool '{self.__name__}' is asynchronous and cannot be called in a synchronous agent run. Use `agent.arun()` instead.")
        
        return self._crewai_tool.run(**kwargs)

    async def __acall__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the CrewAI tool asynchronously."""
        if self.is_async_tool:
            return await self._crewai_tool._run(**kwargs)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._crewai_tool.run(**kwargs))

    def tool_schema(self) -> Dict[str, Any]:
        """Generates the JSON schema for the CrewAI tool."""
        params_schema = {}
        if IS_PYDANTIC_AVAILABLE and hasattr(self._crewai_tool, "args_schema") and self._crewai_tool.args_schema:
            args_schema = self._crewai_tool.args_schema
            
            if not (isinstance(args_schema, type) and issubclass(args_schema, _BaseModel)):
                 raise ValueError("args_schema must be a Pydantic BaseModel class")

            if hasattr(args_schema, "model_json_schema"):
                params_schema = args_schema.model_json_schema()
            elif hasattr(args_schema, "schema"):
                params_schema = args_schema.schema()
            else:
                raise ValueError("args_schema does not have a model_json_schema or schema method.")

            if "title" in params_schema:
                params_schema.pop("title")
        else:
             params_schema = {
                "type": "object",
                "properties": {},
            }

        final_schema = {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.description,
                "parameters": params_schema,
            },
        }
        return {"tool_schema": final_schema, "return_type": self.return_type} 