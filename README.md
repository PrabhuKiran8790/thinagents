# ThinAgents

A lightweight, pluggable AI Agent framework for Python.  
Build LLM-powered agents that can use tools, remember conversations, and connect to external resources with minimal code. ThinAgents leverages `litellm` under the hood for its language model interactions.

[Docs](https://thinagents.vercel.app/)

---

## Installation

```bash
pip install thinagents
```

---

## Basic Usage

Create an agent and interact with an LLM in just a few lines:

```python
from thinagents import Agent

agent = Agent(
    name="Greeting Agent",
    model="openai/gpt-4o-mini",
)

response = await agent.arun("Hello, how are you?")
print(response.content)
```

---

## Tools

Agents can use Python functions as tools to perform actions or fetch data.

```python
from thinagents import Agent

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Agent",
    model="openai/gpt-4o-mini",
    tools=[get_weather],
)

response = await agent.arun("What is the weather in Tokyo?")
print(response.content)
```

---

## Tools with Decorator

For richer metadata and parameter validation, use the `@tool` decorator:

```python
from thinagents import Agent, tool

@tool(name="get_weather")
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Pro",
    model="openai/gpt-4o-mini",
    tools=[get_weather],
)
```

You can also use Pydantic models for parameter schemas:

```python
from pydantic import BaseModel, Field
from thinagents import tool

class MultiplyInputSchema(BaseModel):
    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")

@tool(name="multiply_tool", pydantic_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    return a * b
```

---

## Returning Content and Artifact

Sometimes, a tool should return both a summary (for the LLM) and a large artifact (for downstream use):

```python
from thinagents import tool

@tool(return_type="content_and_artifact")
def summarize_and_return_data(query: str) -> tuple[str, dict]:
    data = {"rows": list(range(10000))}
    summary = f"Found {len(data['rows'])} rows for query: {query}"
    return summary, data

response = await agent.arun("Summarize the data for X")
print(response.content)      # Sent to LLM
print(response.artifact)     # Available for downstream use
```

---

## Async Usage

ThinAgents is async by design. You can stream responses or await the full result:

```python
# Streaming
async for chunk in agent.astream("List files and get weather", conversation_id="1"):
    print(chunk.content, end="", flush=True)

# Or get the full response at once (non-streaming)
response = await agent.arun("List files and get weather", conversation_id="1")
print(response.content)
```

---

## Memory

Agents can remember previous messages and tool results by attaching a memory backend.

```python
from thinagents.memory import InMemoryStore

agent = Agent(
    name="Memory Demo",
    model="openai/gpt-4o-mini",
    memory=InMemoryStore(),  # Fast, in-memory storage
)

conv_id = "demo-1"
print(await agent.arun("Hi, I'm Alice!", conversation_id=conv_id))
print(await agent.arun("What is my name?", conversation_id=conv_id))
# → "Your name is Alice."
```

### Persistent Memory

```python
from thinagents.memory import FileMemory, SQLiteMemory

file_agent = Agent(
    name="File Mem Agent",
    model="openai/gpt-4o-mini",
    memory=FileMemory(storage_dir="./agent_mem"),
)

db_agent = Agent(
    name="SQLite Mem Agent",
    model="openai/gpt-4o-mini",
    memory=SQLiteMemory(db_path="./agent_mem.db"),
)
```

#### Storing Tool Artifacts

Enable artifact storage in memory:

```python
agent = Agent(
    ...,
    memory=InMemoryStore(store_tool_artifacts=True),
)
```

---

## Model Context Protocol (MCP) Integration

Connect your agent to external resources (files, APIs, etc.) using MCP.

```python
agent = Agent(
    name="MCP Agent",
    model="openai/gpt-4o-mini",
    mcp_servers=[
        {
            "transport": "sse",
            "url": "http://localhost:8100/sse"
        },
        {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/path/to/dir"
            ]
        },
    ],
)
```

---

## License

MIT
