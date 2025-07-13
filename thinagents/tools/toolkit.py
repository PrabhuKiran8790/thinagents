"""
Toolkit module for ThinAgents providing organized tool collections.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union
from thinagents.tools.tool import ThinAgentsTool, tool as tool_decorator

logger = logging.getLogger(__name__)


class Toolkit:
    """
    Base class for organizing tools into logical groups with dependency injection support.
    
    Toolkits automatically convert public methods to tools:
    - All public methods (not starting with _) become tools by default
    - Private methods (starting with _) are automatically excluded
    - Use include when you want only specific methods as tools (whitelist)
    - Use exclude when you want to exclude specific methods (blacklist)
    - Use either include OR exclude, not both
    
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
        Initialize the toolset.
        
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
        
        # Discover and convert methods to tools
        self._tools = self._discover_tools()
        
        logger.debug(f"Initialized {self.__class__.__name__} with {len(self._tools)} tools")
    
    def _discover_tools(self) -> List[ThinAgentsTool]:
        """
        Discover methods that should become tools and convert them.
        
        Logic:
        1. Automatically exclude all private methods (starting with _)
        2. Exclude Toolset base class methods
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
            if name in ['get_tools', '_discover_tools']:
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
                    tool_instance.__name__ = f"{self._tool_prefix}_{original_name}"
                tools.append(tool_instance)
            else:
                # Convert regular method to tool
                tool_name = f"{self._tool_prefix}_{name}" if self._tool_prefix else name
                
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
        Get all tools in this toolset.
        
        Returns:
            List of ThinAgentsTool instances
        """
        return self._tools.copy()
    
    def __repr__(self) -> str:
        tool_names = [tool.__name__ for tool in self._tools]
        return f"{self.__class__.__name__}(tools={tool_names}, prefix={self._tool_prefix})" 