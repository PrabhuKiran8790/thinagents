from typing import List, Callable, Any

try:
    from agno.tools import Toolkit as AgnoToolkit  # type: ignore
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    # Create a dummy Toolkit class for type hinting if agno is not installed.
    class AgnoToolkit:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            self.tools: List[Callable] = []


class AgnoIntegrationError(Exception):
    """Custom exception for Agno integration errors."""

    pass


class AgnoTool:
    """
    Adapter to integrate Agno Toolkits with ThinAgents.

    This class wraps an initialized Agno Toolkit instance and provides a simple
    method to extract the individual tool functions in a format that can be
    consumed directly by the thinagents.Agent.
    """

    def __init__(self, toolkit: Any):
        """
        Initializes the adapter with a toolkit instance.

        Args:
            toolkit: An initialized toolkit object that provides a 'tools' attribute as a list of callables.

        Raises:
            AgnoIntegrationError: If an agno toolkit is used but the 'agno' library is not installed.
            AttributeError: If the provided object does not have a 'tools' attribute.
        """
        is_likely_agno = "agno" in str(type(toolkit)).lower()
        if is_likely_agno and not AGNO_AVAILABLE:
            raise AgnoIntegrationError(
                "The 'agno' library is required to use Agno toolkits. Please install it with 'pip install agno'."
            )

        if not hasattr(toolkit, "tools") or not isinstance(
            getattr(toolkit, "tools"), list
        ):
            raise AttributeError(
                "The provided toolkit object does not have a valid 'tools' attribute containing a list of methods."
            )

        self._toolkit = toolkit

    def get_tools(self) -> List[Callable]:
        """
        Returns the list of callable tool functions from the toolkit.
        """
        return self._toolkit.tools 