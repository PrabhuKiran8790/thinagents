"""
MCP (Model Context Protocol) integration for ThinAgents.

This module provides clean integration with MCP servers, allowing agents to use
external tools through the MCP protocol.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Literal, Optional,  TYPE_CHECKING, TypedDict, Tuple, cast
import secrets

if TYPE_CHECKING:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from litellm import experimental_mcp_client

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from litellm import experimental_mcp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Type stubs for when MCP is not available
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    sse_client = None  # type: ignore
    experimental_mcp_client = None  # type: ignore

logger = logging.getLogger(__name__)


# === Transport-agnostic server config ================================
# We now support both stdio-spawned servers **and** remote SSE servers.
#   • For stdio servers – provide `transport="stdio"` (or simply omit it
#     for backward-compat) plus `command` and `args`.
#   • For SSE servers  – provide `transport="sse"` and `url` (and
#     optional `headers`).

# The TypedDict includes optional fields so that one definition covers
# both variants while keeping static type checkers happy.

class MCPServerConfig(TypedDict, total=False):
    """User-facing configuration for a single MCP server.

    Required keys by transport:
      • stdio:  command, args
      • sse:    url
    Optional keys common to both: transport (defaults to "stdio"),
    name, headers (SSE only).
    """

    transport: Literal["stdio", "sse"]  # – default (stdio) handled in normaliser
    name: str

    command: str
    args: List[str]

    url: str
    headers: Dict[str, str]


class MCPServerConfigWithId(TypedDict, total=False):
    """Internal MCP server configuration with required ID."""
    id: str
    transport: Literal["stdio", "sse"]
    name: str
    command: str
    args: List[str]  # stdio-specific
    url: str  # sse-specific
    headers: Dict[str, str]  # sse-specific


class MCPConnectionInfo(TypedDict):
    """Internal MCP connection tracking."""
    session: Any
    connection_context: Any
    session_context: Any
    server_config: MCPServerConfig


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPServerNotAvailableError(MCPError):
    """Raised when MCP dependencies are not installed."""
    pass


def ensure_mcp_available():
    """Ensure MCP dependencies are available."""
    if not MCP_AVAILABLE:
        raise MCPServerNotAvailableError(
            "MCP dependencies not found. Install with: pip install mcp"
        )


class MCPManager:
    """
    Manages MCP server connections and tool loading with automatic cleanup.
    
    Creates fresh connections for each tool loading request to avoid
    issues with reusing connections across different async tasks.
    """
    
    def __init__(self, *, max_parallel_calls: int = 10, failure_threshold: int = 3, backoff_seconds: int = 60):
        self._servers: List[MCPServerConfigWithId] = []
        # Cache for tool schemas/mappings to avoid re-loading on every agent run.
        # The cache is invalidated whenever `add_servers` is called.
        self._tool_cache: Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = None

        # Concurrency control for tool execution
        self._semaphore = asyncio.Semaphore(max_parallel_calls)

        # Circuit-breaker tracking
        self._failure_counts: Dict[str, int] = {}
        self._skip_until: Dict[str, float] = {}

        self._failure_threshold = failure_threshold
        self._backoff_seconds = backoff_seconds
    
    def add_servers(self, servers: List[MCPServerConfigWithId]) -> None:
        """Add MCP servers to be managed."""
        self._servers.extend(servers)
        # Server list changed – invalidate cached tools so they will be re-loaded next time.
        self._tool_cache = None
        logger.debug(f"Added {len(servers)} MCP servers and invalidated tool cache")
    
    async def load_tools(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load tools from all configured MCP servers with fresh connections.
        
        Each call creates new connections to avoid issues with reusing
        connections across different async tasks or agent runs.
        
        Returns:
            Tuple of (tool_schemas, tool_mappings)
        """
        # Return cached schemas/mappings if available to avoid redundant I/O and server startups.
        if self._tool_cache is not None:
            logger.debug("Returning cached MCP tool schemas/mappings")
            return self._tool_cache

        if not self._servers:
            return [], {}

        try:
            # Import MCP dependencies
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.sse import sse_client
            from litellm import experimental_mcp_client
        except ImportError as e:
            raise ImportError(
                f"MCP dependencies not available: {e}. "
                "Install with: pip install mcp"
            ) from e

        all_schemas = []
        all_mappings = {}

        # Helper to pick correct client based on transport
        def connection_cm(s_cfg: MCPServerConfigWithId):  # returns an async CM yielding (read, write)
            transport = s_cfg.get("transport", "stdio")
            if transport == "stdio":
                # Safe: command/args guaranteed by normaliser for stdio transport
                command_val = cast(str, s_cfg["command"])  # type: ignore[index]
                args_val = cast(List[str], s_cfg["args"])  # type: ignore[index]
                server_params_local = StdioServerParameters(
                    command=command_val,
                    args=args_val,
                )
                return stdio_client(server_params_local)
            elif transport == "sse":
                return sse_client(s_cfg["url"], headers=s_cfg.get("headers"))  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown MCP transport '{transport}'.")

        # Create fresh connections for each server with individual context management
        for server_config in self._servers:
            server_id = cast(str, server_config["id"])  # type: ignore[index]

            # Check circuit-breaker skip list
            now = time.time()
            skip_until_ts = self._skip_until.get(server_id)
            if skip_until_ts and now < skip_until_ts:
                logger.warning(
                    f"Skipping MCP server {server_id} due to previous failures. Will retry after {int(skip_until_ts - now)}s."
                )
                continue

            logger.debug(f"Creating fresh connection to MCP server {server_id}")

            try:
                # Open connection using correct transport
                async with connection_cm(server_config) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        logger.debug(f"Initialized fresh MCP session for {server_id}")

                        # Load tools from this server
                        try:
                            tools = await experimental_mcp_client.load_mcp_tools(
                                session=session, 
                                format="openai"
                            )

                            # Add tools directly without prefixing
                            for tool in tools:
                                if isinstance(tool, dict) and "function" in tool:
                                    original_name = tool["function"]["name"]
                                    
                                    # Cast to the expected type to satisfy type checker
                                    tool_dict = cast(Dict[str, Any], tool)
                                    all_schemas.append(tool_dict)

                                    # Create async wrapper that establishes its own connection
                                    def create_tool_wrapper(s_config, orig_name, s_id, sem):
                                        async def tool_wrapper(**kwargs):
                                            # Create fresh connection for each tool call
                                            async with sem:  # Limit concurrency
                                                async with connection_cm(s_config) as (read_inner, write_inner):
                                                    async with ClientSession(read_inner, write_inner) as session_inner:
                                                        await session_inner.initialize()

                                                        # Create tool call dict for MCP
                                                        tool_call_dict = {
                                                            "id": f"call_{orig_name}_{secrets.token_hex(4)}",
                                                            "type": "function",
                                                            "function": {
                                                                "name": orig_name,
                                                                "arguments": __import__('json').dumps(kwargs)
                                                            }
                                                        }

                                                        # Execute the tool call
                                                        result = await experimental_mcp_client.call_openai_tool(
                                                            session=session_inner,
                                                            openai_tool=tool_call_dict  # type: ignore
                                                        )

                                                        # Extract content from result more safely
                                                        if result.content and len(result.content) > 0:
                                                            first_content = result.content[0]
                                                            # Try to get text attribute safely
                                                            text_content = getattr(first_content, "text", None)
                                                            if text_content is not None:
                                                                return text_content
                                                            # Try to get content attribute safely  
                                                            content_attr = getattr(first_content, "content", None)
                                                            return str(content_attr) if content_attr is not None else str(first_content)
                                                        else:
                                                            return f"Tool {orig_name} executed successfully"

                                        # Mark as async tool
                                        tool_wrapper.is_async_tool = True
                                        tool_wrapper.__name__ = original_name

                                        return tool_wrapper

                                    # Create the wrapper using original name
                                    wrapper = create_tool_wrapper(server_config, original_name, server_id, self._semaphore)

                                    # Detect duplicate tool names across MCP servers to avoid silent collisions.
                                    if original_name in all_mappings:
                                        raise ValueError(
                                            f"Duplicate MCP tool name '{original_name}' detected while loading from server {server_id}. "
                                            "Ensure tool names are unique across MCP servers or prefix them explicitly."
                                        )

                                    all_mappings[original_name] = wrapper

                            logger.info(f"Loaded {len(tools)} tools from MCP server {server_id}")

                            # Reset failure count on success
                            self._failure_counts.pop(server_id, None)
                            self._skip_until.pop(server_id, None)

                        except Exception as e:
                            logger.error(f"Failed to load tools from MCP server {server_id}: {e}", exc_info=True)

                            # Increment failure count and potentially activate back-off
                            self._failure_counts[server_id] = self._failure_counts.get(server_id, 0) + 1
                            if self._failure_counts[server_id] >= self._failure_threshold:
                                self._skip_until[server_id] = time.time() + self._backoff_seconds
                                logger.warning(
                                    f"MCP server {server_id} failed {self._failure_counts[server_id]} times — backing off for {self._backoff_seconds}s."
                                )
                            continue
                                            
                                        # Session and connection are automatically cleaned up by the context managers

            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}", exc_info=True)

                # Increment failure count and maybe back-off
                self._failure_counts[server_id] = self._failure_counts.get(server_id, 0) + 1
                if self._failure_counts[server_id] >= self._failure_threshold:
                    self._skip_until[server_id] = time.time() + self._backoff_seconds
                    logger.warning(
                        f"MCP server {server_id} failed {self._failure_counts[server_id]} times — backing off for {self._backoff_seconds}s."
                    )
                continue

        # Cache the successfully loaded schemas/mappings for future calls.
        self._tool_cache = (all_schemas, all_mappings)
        return all_schemas, all_mappings
    
    async def cleanup(self) -> None:
        """
        Clean up MCP connections.
        
        This method is provided for compatibility but connections are now
        managed automatically via context managers in each load_tools call.
        """
        # No persistent connections to clean up
        logger.debug("MCP cleanup called - using fresh connections, no cleanup needed")
    
    def __del__(self):
        """Cleanup when manager is garbage collected."""
        # No persistent connections to clean up
        pass


def normalize_mcp_servers(servers: Optional[List[MCPServerConfig]]) -> List[MCPServerConfigWithId]:
    """
    Normalize MCP server configurations by adding unique IDs.
    
    Args:
        servers: List of MCP server configurations or None
        
    Returns:
        List of normalized server configurations with IDs
    """
    if not servers:
        return []
    
    normalized: List[MCPServerConfigWithId] = []

    for server in servers:
        transport = server.get("transport", "stdio")

        server_id = f"mcp_{secrets.token_hex(4)}"

        if transport == "stdio":
            command = server.get("command")
            args = server.get("args")
            if command is None or args is None:
                raise ValueError("stdio MCP server config must include 'command' and 'args'.")

            normalized_server: MCPServerConfigWithId = {
                "id": server_id,
                "transport": "stdio",
                "name": server.get("name", ""),
                "command": command,
                "args": args,
            }

        elif transport == "sse":
            url = server.get("url")
            if url is None:
                raise ValueError("sse MCP server config must include 'url'.")

            normalized_server = {
                "id": server_id,
                "transport": "sse",
                "name": server.get("name", ""),
                "url": url,
            }

            # Optional headers
            headers = server.get("headers")
            if headers is not None:
                normalized_server["headers"] = headers  # type: ignore[index]

        else:
            raise ValueError(f"Unknown MCP transport '{transport}'.")

        normalized.append(normalized_server)

    return normalized 