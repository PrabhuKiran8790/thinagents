"""
Toolkit module for ThinAgents providing organized tool collections.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union
from thinagents.tools.tool import ThinAgentsTool, tool as tool_decorator, sanitize_function_name, FunctionNameSanitizationError

logger = logging.getLogger(__name__)


class Toolkit:
    """
    Base class for organizing tools into logical groups with toolkit context support.
    
    Toolkits automatically convert public methods to tools:
    - All public methods (not starting with _) become tools by default
    - Private methods (starting with _) are automatically excluded
    - Use include when you want only specific methods as tools (whitelist)
    - Use exclude when you want to exclude specific methods (blacklist)
    - Use either include OR exclude, not both
    
    IMPORTANT: If implementing toolkit_context() method, it MUST be the last method 
    defined in your toolkit class to ensure all other methods are available when it executes.
    
    Class Attributes:
        include: Optional list of method names to expose as tools (whitelist).
                If specified, ONLY these methods become tools.
        exclude: Optional list of method names to exclude from becoming tools (blacklist).
                Applied to all public methods.
        tool_prefix: Optional prefix to add to all tool names.
    """
    
    # Class-level configuration attributes
    include: Optional[Union[List[str], Set[str]]] = None
    exclude: Optional[Union[List[str], Set[str]]] = None
    tool_prefix: Optional[str] = None
    
    def __init__(self):
        """
        Initialize the toolkit.
        
        Uses class-level attributes for configuration:
        - include: List/set of method names to include as tools. If None, all public methods are included.
        - exclude: List/set of method names to exclude from becoming tools.
        - tool_prefix: Prefix to add to all tool names (e.g., "calc" -> "calc_add", "calc_multiply").
        """
        # Validate that both include and exclude are not used together
        if self.include is not None and self.exclude is not None:
            raise ValueError("Cannot use both 'include' and 'exclude' together. Use either include (whitelist) or exclude (blacklist).")
        
        # Convert class attributes to instance attributes for processing
        self._include = set(self.include) if self.include else None
        self._exclude = set(self.exclude) if self.exclude else None
        self._tool_prefix = self.tool_prefix
        
        # Validate toolkit_context() method placement (optional but helpful)
        self._validate_toolkit_context_placement()
        
        # Discover and convert methods to tools
        self._tools = self._discover_tools()
        
        # Generate toolkit context
        self._context_string = self._build_toolkit_context()
        
        logger.debug(f"Initialized {self.__class__.__name__} with {len(self._tools)} tools")
    
    def _validate_toolkit_context_placement(self) -> None:
        """
        Validate that toolkit_context() is the last defined method.
        This helps catch potential execution order issues early.
        """
        if not hasattr(self, 'toolkit_context'):
            # No toolkit_context() method defined, which is fine
            return

        try:
            # Get all methods defined in this class (not inherited)
            class_methods = []
            class_methods.extend(
                (name, inspect.getsourcelines(method)[1])
                for name, method in inspect.getmembers(
                    self.__class__, predicate=inspect.isfunction
                )
                if name in self.__class__.__dict__
            )
            # Sort by line number to get definition order
            class_methods.sort(key=lambda x: x[1])

            if class_methods and class_methods[-1][0] != 'toolkit_context':
                logger.warning(
                    f"toolkit_context() should be the last method defined in {self.__class__.__name__} "
                    f"to ensure all methods are available during execution. "
                    f"Currently last method is: {class_methods[-1][0]}"
                )
        except Exception as e:
            # Don't fail initialization if validation fails
            logger.debug(f"Could not validate toolkit_context() placement: {e}")
    
    def _build_toolkit_context(self) -> Optional[str]:
        """
        Execute toolkit_context() method to get the context string.
        """
        if not hasattr(self, 'toolkit_context') or not callable(getattr(self, 'toolkit_context')):
            return None
        
        try:
            context = self.toolkit_context()
            if context and isinstance(context, str):
                logger.debug(f"Generated toolkit context for {self.__class__.__name__}")
                return context
            else:
                logger.warning(f"toolkit_context() in {self.__class__.__name__} did not return a string")
                return None
        except Exception as e:
            logger.error(f"Failed to execute toolkit_context() in {self.__class__.__name__}: {e}")
            return None
    
    def get_toolkit_context(self) -> Optional[str]:
        """Get the generated toolkit context string."""
        return self._context_string
    
    def toolkit_context(self) -> str:
        """
        Override this method to provide toolkit-specific context.
        
        This method should be the LAST method defined in your toolkit class
        to ensure all other methods are available when it executes.
        
        Returns:
            String to be appended to the agent's system prompt
        """
        return ""

    def _discover_tools(self) -> List[ThinAgentsTool]:
        """
        Discover methods that should become tools and convert them.
        
        Logic:
        1. Automatically exclude all private methods (starting with _)
        2. Exclude Toolkit base class methods
        3. If include is specified: Only include those methods (whitelist)
        4. If exclude is specified: Exclude those methods from all public methods (blacklist)
        5. Convert remaining methods to tools
        
        Returns:
            List of ThinAgentsTool instances
        """
        tools = []
        
        # Common Python internals and artifacts to exclude
        EXCLUDED_INTERNALS = {
            'args', 'kwargs', 'self', '__class__', '__dict__', '__doc__', '__module__', 
            '__weakref__', '__annotations__', '__qualname__', '__name__'
        }
        
        # Get all methods of the class
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            # Automatically exclude private methods and special methods
            if name.startswith('_'):
                continue
            
            # Skip common Python internals and artifacts
            if name in EXCLUDED_INTERNALS:
                continue
                
            # Skip methods that are part of the Toolkit base class
            if name in ['get_tools', '_discover_tools', 'get_toolkit_context', 'toolkit_context']:
                continue
            
            # Additional validation: ensure this is actually a method defined on the class
            # and not some artifact from introspection
            try:
                # Check if this is actually a bound method of our class
                if not hasattr(self.__class__, name):
                    logger.warning(f"Skipping '{name}' - not found on class {self.__class__.__name__}")
                    continue
                    
                # Get the unbound method from the class
                class_method = getattr(self.__class__, name)
                if not callable(class_method):
                    logger.warning(f"Skipping '{name}' - not callable on class {self.__class__.__name__}")
                    continue
                    
                # Ensure it's actually a method (not a property or descriptor)
                if not inspect.isfunction(class_method) and not inspect.ismethod(class_method):
                    logger.warning(f"Skipping '{name}' - not a function or method on class {self.__class__.__name__}")
                    continue
            except Exception as e:
                logger.warning(f"Skipping '{name}' due to introspection error: {e}")
                continue
            
            # If include is specified, only include those methods (whitelist)
            if self._include is not None and name not in self._include:
                continue
            
            # If exclude is specified, exclude those methods (blacklist)
            if self._exclude is not None and name in self._exclude:
                continue
            
            # At this point, the method should become a tool
            # Check if method is already decorated with @tool
            if hasattr(method, 'tool_schema'):
                # Method is already a tool, just update the name with prefix if needed
                tool_instance = method
                if self._tool_prefix:
                    original_name = tool_instance.__name__
                    try:
                        tool_instance.__name__ = sanitize_function_name(f"{self._tool_prefix}_{original_name}")
                    except FunctionNameSanitizationError as e:
                        logger.error(f"Failed to sanitize tool name '{self._tool_prefix}_{original_name}': {e}")
                        raise ValueError(f"Cannot create valid tool name for method '{original_name}' with prefix '{self._tool_prefix}': {e}") from e
                tools.append(tool_instance)
            else:
                # Convert regular method to tool
                raw_tool_name = f"{self._tool_prefix}_{name}" if self._tool_prefix else name
                try:
                    tool_name = sanitize_function_name(raw_tool_name)
                except FunctionNameSanitizationError as e:
                    logger.error(f"Failed to sanitize tool name '{raw_tool_name}': {e}")
                    raise ValueError(f"Cannot create valid tool name for method '{name}': {e}") from e
                
                # Create a wrapper that binds the method to self
                def create_tool_wrapper(method_ref, tool_name):
                    # Get the method signature to understand what parameters it accepts
                    sig = inspect.signature(method_ref)
                    
                    def wrapper(*args, **kwargs):
                        # Filter kwargs to only include parameters the method accepts
                        filtered_kwargs = {}
                        for param_name, param in sig.parameters.items():
                            if param_name in kwargs:
                                filtered_kwargs[param_name] = kwargs[param_name]
                        
                        # Check if method accepts **kwargs
                        accepts_var_keyword = any(
                            param.kind == inspect.Parameter.VAR_KEYWORD 
                            for param in sig.parameters.values()
                        )
                        
                        if accepts_var_keyword:
                            # Method accepts **kwargs, pass all kwargs
                            return method_ref(*args, **kwargs)
                        else:
                            # Method doesn't accept **kwargs, only pass filtered kwargs
                            return method_ref(*args, **filtered_kwargs)
                    
                    wrapper.__name__ = tool_name
                    wrapper.__doc__ = method_ref.__doc__
                    wrapper.__annotations__ = getattr(method_ref, '__annotations__', {})
                    
                    return wrapper
                
                wrapper = create_tool_wrapper(method, tool_name)
                tool_instance = tool_decorator(wrapper)
                tools.append(tool_instance)
        
        return tools
    
    def get_tools(self) -> List[ThinAgentsTool]:
        """
        Get all tools in this toolkit.
        
        Returns:
            List of ThinAgentsTool instances
        """
        return self._tools.copy()
    
    def get_tools_info(self) -> str:
        """
        Get formatted information about all tools in this toolkit.
        
        Returns:
            A formatted string with numbered list of tools and their descriptions.
        """
        tools = self.get_tools()
        if not tools:
            return ""
        
        tool_info_lines = []
        for i, tool in enumerate(tools, 1):
            try:
                schema_data = tool.tool_schema()
                # Extract the actual OpenAI tool schema
                if isinstance(schema_data, dict) and "tool_schema" in schema_data:
                    actual_schema = schema_data["tool_schema"]
                else:
                    actual_schema = schema_data
                
                # Get tool name and description from schema
                if isinstance(actual_schema, dict) and "function" in actual_schema:
                    function_info = actual_schema["function"]
                    tool_name = function_info.get("name", tool.__name__)
                    full_description = function_info.get("description", "No description available")
                    
                    # Extract only the main description, ignore Args: sections
                    description = full_description.split('\n')[0].split('Args:')[0].strip() or "No description available"
                else:
                    tool_name = tool.__name__
                    description = "No description available"
                
                tool_info_lines.append(f"{i}. {tool_name}: {description}")
                
            except Exception as e:
                logger.warning(f"Failed to get schema for tool {tool.__name__}: {e}")
                tool_info_lines.append(f"{i}. {tool.__name__}: No description available")
        
        return "\n".join(tool_info_lines)

    def __repr__(self) -> str:
        tool_names = [tool.__name__ for tool in self._tools]
        return f"{self.__class__.__name__}(tools={tool_names}, prefix={self._tool_prefix})" 