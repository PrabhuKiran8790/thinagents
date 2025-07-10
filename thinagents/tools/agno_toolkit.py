from typing import List, Callable, Any

try:
    from agno.tools import Toolkit as AgnoToolkit
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    # Create a dummy Toolkit class for type hinting if agno is not installed.
    class AgnoToolkit: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            self.tools: List[Callable] = []

class AgnoIntegrationError(Exception):
    """Custom exception for Agno integration errors."""
    pass

class AgnoToolkitAdapter:
    """
    Adapter to integrate Agno Toolkits with ThinAgents.

    This class wraps an initialized Agno Toolkit instance and provides a simple
    method to extract the individual tool functions in a format that can be
    consumed directly by the thinagents.Agent.
    """
    def __init__(self, toolkit: AgnoToolkit):
        """
        Initializes the adapter with an Agno Toolkit instance.

        Args:
            toolkit: An initialized instance of a class that inherits from agno.tools.Toolkit.

        Raises:
            AgnoIntegrationError: If the 'agno' library is not installed.
            TypeError: If the provided object is not an instance of agno.tools.Toolkit.
        """
        if not AGNO_AVAILABLE:
            raise AgnoIntegrationError("The 'agno' library is required to use Agno toolkits. Please install it with 'pip install agno'.")

        if not isinstance(toolkit, AgnoToolkit):
            raise TypeError(f"Expected an instance of agno.tools.Toolkit, but got {type(toolkit).__name__}.")

        if not hasattr(toolkit, "tools") or not isinstance(getattr(toolkit, "tools"), list):
            raise AttributeError("The provided toolkit object does not have a valid 'tools' attribute containing a list of methods.")
            
        self._toolkit = toolkit

    def get_tools(self) -> List[Callable]:
        """
        Returns the list of callable tool functions from the Agno Toolkit.
        """
        return self._toolkit.tools 