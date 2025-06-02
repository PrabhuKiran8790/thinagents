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
        self._tool_cache: Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = None

        self._semaphore = asyncio.Semaphore(max_parallel_calls)
        """
        We use a semaphore to limit the number of concurrent calls to the MCP servers.
        This is to avoid overwhelming the servers and to avoid rate limiting.
        """

        self._failure_counts: Dict[str, int] = {}
        self._skip_until: Dict[str, float] = {}

        self._failure_threshold = failure_threshold
        self._backoff_seconds = backoff_seconds
    
    def add_servers(self, servers: List[MCPServerConfigWithId]) -> None:
        """Add MCP servers to be managed."""
        self._servers.extend(servers)
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
        ensure_mcp_available()

        from mcp import ClientSession, StdioServerParameters  # type: ignore
        from mcp.client.stdio import stdio_client  # type: ignore
        from mcp.client.sse import sse_client  # type: ignore
        from litellm import experimental_mcp_client  # type: ignore

        if self._tool_cache is not None:
            logger.debug("Returning cached MCP tool schemas/mappings")
            return self._tool_cache

        if not self._servers:
            return [], {}

        all_schemas: List[Dict[str, Any]] = []
        all_mappings: Dict[str, Any] = {}

        def connection_cm(s_cfg: MCPServerConfigWithId):  # returns an async CM yielding (read, write)
            transport = s_cfg.get("transport", "stdio")
            if transport == "stdio":
                command_val = cast(str, s_cfg["command"])  # type: ignore[index]
                args_val = cast(List[str], s_cfg["args"])  # type: ignore[index]
                server_params_local = StdioServerParameters(
                    command=command_val,
                    args=args_val,
                )
                return stdio_client(server_params_local)
            elif transport == "sse":
                return sse_client(s_cfg["url"], headers=s_cfg.get("headers"))  # type: ignore[arg-type]
            raise ValueError(f"Unknown MCP transport '{transport}'.")

        for server_config in self._servers:
            server_id = cast(str, server_config["id"])  # type: ignore[index]

            now = time.time()
            skip_until_ts = self._skip_until.get(server_id)
            if skip_until_ts and now < skip_until_ts:
                logger.warning(
                    f"Skipping MCP server {server_id} due to previous failures. Will retry after {int(skip_until_ts - now)}s."
                )
                continue

            logger.debug(f"Creating fresh connection to MCP server {server_id}")

            try:
                async with connection_cm(server_config) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        logger.debug(f"Initialized fresh MCP session for {server_id}")

                        try:
                            tools = await experimental_mcp_client.load_mcp_tools(
                                session=session, 
                                format="openai"
                            )

                            for tool in tools:
                                if isinstance(tool, dict) and "function" in tool:
                                    original_name = tool["function"]["name"]
                                    
                                    tool_dict = cast(Dict[str, Any], tool)
                                    all_schemas.append(tool_dict)

                                    def create_tool_wrapper(s_config, orig_name, sem):
                                        async def tool_wrapper(*, _s_config=s_config, _orig_name=orig_name, _sem=sem, **kwargs):
                                            async with _sem:
                                                async with connection_cm(_s_config) as (read_inner, write_inner):
                                                    async with ClientSession(read_inner, write_inner) as session_inner:
                                                        await session_inner.initialize()

                                                        tool_call_dict = {
                                                            "id": f"call_{_orig_name}_{secrets.token_hex(4)}",
                                                            "type": "function",
                                                            "function": {
                                                                "name": _orig_name,
                                                                "arguments": __import__('json').dumps(kwargs)
                                                            }
                                                        }

                                                        result = await experimental_mcp_client.call_openai_tool(
                                                            session=session_inner,
                                                            openai_tool=tool_call_dict  # type: ignore
                                                        )

                                                        if result.content:
                                                            first = result.content[0]
                                                            txt = getattr(first, "text", None)
                                                            if txt is not None:
                                                                return txt
                                                            cont = getattr(first, "content", None)
                                                            return str(cont) if cont is not None else str(first)
                                                        return f"Tool {_orig_name} executed successfully"

                                        tool_wrapper.is_async_tool = True  # type: ignore[attr-defined]
                                        tool_wrapper.__name__ = orig_name
                                        return tool_wrapper

                                    wrapper = create_tool_wrapper(server_config, original_name, self._semaphore)

                                    if original_name in all_mappings:
                                        raise ValueError(
                                            f"Duplicate MCP tool name '{original_name}' detected while loading from server {server_id}. "
                                            "Ensure tool names are unique across MCP servers or prefix them explicitly."
                                        )

                                    all_mappings[original_name] = wrapper

                            logger.info(f"Loaded {len(tools)} tools from MCP server {server_id}")

                            self._failure_counts.pop(server_id, None)
                            self._skip_until.pop(server_id, None)

                        except Exception as e:
                            logger.warning(
                                "Failed to load tools from MCP server %s (config=%s): %s. Skipping but continuing with other servers.",
                                server_id,
                                server_config,
                                e,
                            )

            except Exception as e:
                # Connection failure – log as warning without full traceback to keep logs clean.
                logger.warning(
                    "Failed to connect to MCP server %s (config=%s): %s. Skipping this server.",
                    server_id,
                    server_config,
                    e,
                )

                # Increment failure count and maybe back-off
                self._failure_counts[server_id] = self._failure_counts.get(server_id, 0) + 1
                if self._failure_counts[server_id] >= self._failure_threshold:
                    self._skip_until[server_id] = time.time() + self._backoff_seconds
                    logger.warning(
                        f"MCP server {server_id} failed {self._failure_counts[server_id]} times — backing off for {self._backoff_seconds}s."
                    )
                continue

        self._tool_cache = (all_schemas, all_mappings)
        return all_schemas, all_mappings
    
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