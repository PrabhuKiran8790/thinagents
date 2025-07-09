"""ThinAgents - A lightweight library for building AI agents

ThinAgents is a minimalist framework for building and deploying AI agents powered by
popular language models. Key features:

- Integration with litellm (https://www.litellm.ai) for 100+ LLM APIs
- Support for major providers like OpenAI, Anthropic, Google and more
- Simple, straightforward API focused on core agent functionality
- Small footprint and easy deployment

Build powerful AI agents quickly with your preferred model provider using ThinAgents'
streamlined framework.
"""

from thinagents.core.agent import Agent
from thinagents.utils.prompts import PromptConfig
from thinagents.tools.tool import tool
from thinagents.core.response_models import ThinagentResponse
from thinagents.core.mcp import MCPServerConfig


__all__ = ["Agent", "PromptConfig", "tool", "ThinagentResponse", "MCPServerConfig"]
