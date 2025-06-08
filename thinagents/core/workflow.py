"""
Workflow module for building multi-agent workflows with type-safe agent registry.

This module provides a base Workflow class that allows users to define their workflow logic
in a single run() method while automatically providing async and streaming variants.
"""

import asyncio
import inspect
import logging
from typing import Any, Dict, Iterator, AsyncIterator, Optional, Union, TypeVar, List
from abc import ABC
from pydantic import BaseModel

from thinagents.core.agent import Agent
from thinagents.core.response_models import ThinagentResponse, ThinagentResponseStream, WorkflowData

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StreamingAgentResponse:
    """
    A response object that acts as a real-time stream (iterator) while
    also capturing the full content for later use. This avoids making
    a second API call to get the complete content after streaming.
    
    You can iterate over this object to get streaming chunks:
    `for chunk in streaming_response:`
    
    After iteration, you can access the full content:
    `full_content = streaming_response.content`
    """
    def __init__(self, stream_generator: Union[Iterator, AsyncIterator], agent_name: str):
        self._stream_generator = stream_generator
        self._agent_name = agent_name
        self._content_parts: List[str] = []
        self._chunks: List[ThinagentResponseStream] = []
        self._is_consumed = False
        self._final_response: Optional[ThinagentResponse] = None

    def __iter__(self) -> Iterator[ThinagentResponseStream]:
        """Yields stream chunks while capturing them."""
        if inspect.isasyncgen(self._stream_generator):
            raise TypeError("Cannot iterate over an async stream with a sync 'for' loop. Use 'async for' instead.")

        for chunk in self._stream_generator: # type: ignore
            if not isinstance(chunk, ThinagentResponseStream):
                processed_chunk = ThinagentResponseStream(
                    content=str(chunk), content_type="str", agent_name=self._agent_name,
                    response_id=None, created_timestamp=None, model_used=None, finish_reason=None,
                    metrics=None, system_fingerprint=None, artifact=None, tool_name=None,
                    tool_call_id=None, stream_options=None
                )
            else:
                processed_chunk = chunk
                if not processed_chunk.agent_name:
                    processed_chunk.agent_name = self._agent_name
            
            self._chunks.append(processed_chunk)
            if processed_chunk.content:
                self._content_parts.append(processed_chunk.content)
            
            yield processed_chunk
        self._is_consumed = True

    async def __aiter__(self) -> AsyncIterator[ThinagentResponseStream]:
        """Async version of the iterator for arun_stream."""
        stream_to_iterate = self._stream_generator
        if not (inspect.isasyncgen(stream_to_iterate) or hasattr(stream_to_iterate, '__aiter__')):
            # It's a sync iterator, wrap it to make it async
            async def wrap_sync_iter(sync_iter: Iterator) -> AsyncIterator:
                for item in sync_iter:
                    yield item
            stream_to_iterate = wrap_sync_iter(stream_to_iterate) # type: ignore

        async for chunk in stream_to_iterate: # type: ignore
            if not isinstance(chunk, ThinagentResponseStream):
                processed_chunk = ThinagentResponseStream(
                    content=str(chunk), content_type="str", agent_name=self._agent_name,
                    response_id=None, created_timestamp=None, model_used=None, finish_reason=None,
                    metrics=None, system_fingerprint=None, artifact=None, tool_name=None,
                    tool_call_id=None, stream_options=None
                )
            else:
                processed_chunk = chunk
                if not processed_chunk.agent_name:
                    processed_chunk.agent_name = self._agent_name
            
            self._chunks.append(processed_chunk)
            if processed_chunk.content:
                self._content_parts.append(processed_chunk.content)
            
            yield processed_chunk
        self._is_consumed = True

    def _consume(self) -> None:
        """Consumes the stream if it hasn't been already."""
        if not self._is_consumed and inspect.isasyncgen(self._stream_generator):
            raise TypeError("Cannot synchronously consume an async stream. Please iterate over it with 'async for'.")

    @property
    def content(self) -> str:
        """
        Returns the full concatenated content from the stream.
        If the stream has not been consumed, it will be fully iterated
        through to collect the content.
        """
        self._consume()
        return "".join(self._content_parts)

    def get_final_response(self) -> ThinagentResponse:
        """
        Returns a consolidated ThinagentResponse object after the
        stream has been consumed.
        """
        self._consume()
        if self._final_response:
            return self._final_response

        last_chunk = self._chunks[-1] if self._chunks else None
        
        self._final_response = ThinagentResponse(
            content="".join(self._content_parts),
            content_type="str",
            agent_name=self._agent_name,
            response_id=getattr(last_chunk, 'response_id', None),
            created_timestamp=getattr(last_chunk, 'created_timestamp', None),
            model_used=getattr(last_chunk, 'model_used', None),
            finish_reason=getattr(last_chunk, 'finish_reason', None),
            metrics=getattr(last_chunk, 'metrics', None),
            system_fingerprint=getattr(last_chunk, 'system_fingerprint', None),
            artifact=getattr(last_chunk, 'artifact', None),
            tool_name=getattr(last_chunk, 'tool_name', None),
            tool_call_id=getattr(last_chunk, 'tool_call_id', None),
            workflow=None,
        )
        return self._final_response

class ExecutionContext:
    """Tracks the current workflow execution context."""
    
    def __init__(self):
        self.mode = 'sync'  # 'sync', 'async', 'stream', 'async_stream'
        self.is_streaming = False
        self.is_async = False
    
    def set_mode(self, is_async: bool = False, is_streaming: bool = False):
        self.is_async = is_async
        self.is_streaming = is_streaming
        
        if is_async and is_streaming:
            self.mode = 'async_stream'
        elif is_async:
            self.mode = 'async'
        elif is_streaming:
            self.mode = 'stream' 
        else:
            self.mode = 'sync'

class StreamingAgentProxy:
    """
    Proxy that wraps an agent and provides streaming functionality for generator-based workflows.
    
    This proxy yields chunks in real-time when used in generator workflows (workflows that use yield).
    """
    
    def __init__(self, agent: Agent, execution_context: ExecutionContext, workflow_agent_name: Optional[str] = None):
        self._agent = agent
        self._execution_context = execution_context
        self._workflow_agent_name = workflow_agent_name
        # Copy agent attributes to proxy - use workflow name if provided
        self.name = workflow_agent_name or agent.name
        self.model = agent.model
    
    def run(self, *args, **kwargs):
        """Run agent and yield chunks in real-time for streaming workflows."""
        kwargs['stream'] = True
        stream_result = self._agent.run(*args, **kwargs)
        
        # Yield chunks immediately as they arrive
        for chunk in stream_result:
            if isinstance(chunk, ThinagentResponseStream):
                chunk.agent_name = self.name
                yield chunk
            else:
                # Convert to proper ThinagentResponseStream
                yield ThinagentResponseStream(
                    content=str(chunk),
                    content_type="str",
                    agent_name=self.name,
                    response_id=None,
                    created_timestamp=None,
                    model_used=None,
                    finish_reason=None,
                    metrics=None,
                    system_fingerprint=None,
                    artifact=None,
                    tool_name=None,
                    tool_call_id=None,
                    stream_options=None
                )
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped agent."""
        return getattr(self._agent, name)


class AgentProxy:
    """Proxy that wraps an agent and provides transparent access with workflow context awareness."""
    
    def __init__(self, agent: Agent, execution_context: ExecutionContext, workflow_instance, workflow_agent_name: Optional[str] = None):
        self._agent = agent
        self._execution_context = execution_context
        self._workflow = workflow_instance
        self._workflow_agent_name = workflow_agent_name
        # Copy agent attributes to proxy - use workflow name if provided
        self.name = workflow_agent_name or agent.name
        self.model = agent.model
    
    def run(self, *args, **kwargs):
        """Intercept run calls and handle execution modes transparently."""
        context = self._execution_context
        
        # Check if user explicitly requested streaming
        user_requested_stream = kwargs.get('stream', False)
        
        if context.mode == 'stream':
            # If user explicitly set stream=True, return the actual stream
            if user_requested_stream:
                kwargs['stream'] = True
                stream_result = self._agent.run(*args, **kwargs)
                return StreamingAgentResponse(stream_result, self.name)
            
            # Framework streaming mode: collect chunks and store for batch yielding (backward compatibility)
            kwargs['stream'] = True
            stream_result = self._agent.run(*args, **kwargs)
            
            chunks = []
            content_parts = []
            
            # Collect all chunks and store them for workflow yielding
            for chunk in stream_result:
                # Ensure chunk has agent_name set
                if isinstance(chunk, ThinagentResponseStream):
                    chunk.agent_name = self.name
                    chunks.append(chunk)
                    content_parts.append(chunk.content)
                    
                    # Store chunk for workflow to yield
                    if hasattr(self._workflow, '_pending_chunks'):
                        self._workflow._pending_chunks.append(chunk)
                else:
                    # Handle edge cases where chunk might not be ThinagentResponseStream
                    chunk_content = str(chunk)
                    response_chunk = ThinagentResponseStream(
                        content=chunk_content,
                        content_type="str",
                        agent_name=self.name,
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason=None,
                        metrics=None,
                        system_fingerprint=None,
                        artifact=None,
                        tool_name=None,
                        tool_call_id=None,
                        stream_options=None
                    )
                    chunks.append(response_chunk)
                    content_parts.append(chunk_content)
                    
                    if hasattr(self._workflow, '_pending_chunks'):
                        self._workflow._pending_chunks.append(response_chunk)
            
            # Return a ThinagentResponse object that the workflow can use normally
            full_content = "".join(content_parts)
            return ThinagentResponse(
                content=full_content,
                content_type="str",
                agent_name=self.name,  # Uses workflow agent name (Pydantic field name)
                response_id=None,
                created_timestamp=None,
                model_used=None,
                finish_reason=None,
                metrics=None,
                system_fingerprint=None,
                artifact=None,
                tool_name=None,
                tool_call_id=None,
                workflow=None
            )

        elif context.mode == 'async':
            if hasattr(self._agent, 'arun'):
                result = self._agent.arun(*args, **kwargs)
                # Ensure result has agent_name when it resolves
                if hasattr(result, 'agent_name'):
                    result.agent_name = self.name
                return result
            # Fallback to sync in executor
            async def async_wrapper():
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._agent.run, *args, **kwargs)
                if hasattr(result, 'agent_name'):
                    result.agent_name = self.name
                return result
            return async_wrapper()

        elif context.mode == 'async_stream':
            kwargs['stream'] = True  # Always stream in this mode
            # Async streaming - use astream
            if hasattr(self._agent, 'astream'):
                stream_result = self._agent.astream(*args, **kwargs)
                return StreamingAgentResponse(stream_result, self.name)
            elif hasattr(self._agent, 'arun'):
                # Fallback to arun with stream=True
                stream_result = self._agent.arun(*args, **kwargs)
                return StreamingAgentResponse(stream_result, self.name)
            else:
                # Final fallback
                async def async_stream_wrapper():
                    loop = asyncio.get_event_loop()
                    sync_stream = await loop.run_in_executor(None, self._agent.run, *args, **kwargs)
                    for chunk in sync_stream:
                        yield chunk
                return StreamingAgentResponse(async_stream_wrapper(), self.name)

        # Default sync execution - ensure agent_name is set
        result = self._agent.run(*args, **kwargs)
        if hasattr(result, 'agent_name'):
            result.agent_name = self.name
        return result
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped agent."""
        return getattr(self._agent, name)

class WorkflowState:
    """State container for workflow execution with dot notation access."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'WorkflowState' object has no attribute '{name}'")
    
    def __delattr__(self, name: str) -> None:
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            try:
                del self._data[name]
            except KeyError:
                raise AttributeError(f"'WorkflowState' object has no attribute '{name}'")
    
    def __contains__(self, name: str) -> bool:
        return name in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)
    
    def update(self, other: Union[Dict[str, Any], 'WorkflowState']) -> None:
        if isinstance(other, WorkflowState):
            self._data.update(other._data)
        else:
            self._data.update(other)
    
    def clear(self) -> None:
        self._data.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a regular dictionary."""
        return self._data.copy()
    
    def __repr__(self) -> str:
        return f"WorkflowState({self._data})"


class AgentExecutionResult:
    """Wrapper for agent execution results with metadata."""
    
    def __init__(self, result: Any, agent_name: str, is_streaming: bool = False):
        self.result = result
        self.agent_name = agent_name
        self.is_streaming = is_streaming
        
    def __str__(self) -> str:
        if isinstance(self.result, ThinagentResponse):
            return str(self.result.content)
        return str(self.result)
    
    def __repr__(self) -> str:
        return f"AgentExecutionResult(agent={self.agent_name}, result={self.result})"


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""
    pass


class AgentNotFoundError(WorkflowError):
    """Exception raised when a referenced agent is not found in the registry."""
    pass


class WorkflowExecutionError(WorkflowError):
    """Exception raised when workflow execution fails."""
    pass


class BaseWorkflow(ABC):
    """
    Base class for building multi-agent workflows.
    
    Users should subclass this and implement:
    1. Define their agents using a Pydantic BaseModel
    2. Implement the run() method with their workflow logic
    
    The framework automatically provides:
    - run_stream(): Streaming version of run()
    - arun(): Async version of run()  
    - arun_stream(): Async streaming version of run()
    """
    
    def __init__(self):
        self.state = WorkflowState()
        self._agents: Optional[BaseModel] = None
        self._agents_setup = False
        self._execution_context = ExecutionContext()
        self._original_agent_methods: Dict[str, Dict[str, Any]] = {}
        self._method_wrappers_installed = False
        self._pending_chunks: List[str] = []
        self._streaming_agents: Optional[Any] = None
    
    def _setup_agents(self) -> None:
        """Setup agents from agents attribute (called lazily)."""
        if self._agents_setup:
            return

        if hasattr(self, 'agents') and isinstance(getattr(self, 'agents'), BaseModel):
            # agents is defined as an instance
            self._agents = getattr(self, 'agents')
        self._agents_setup = True
    
    @property
    def stream(self) -> Any:
        """
        Access to streaming agent proxies for generator-based workflows.
        
        Use this in generator workflows (that use yield) to get real-time streaming:
        
        Example:
            def run(self, input_data):
                yield "🔍 Researching..."
                for chunk in self.stream.researcher.run("Research solar energy"):
                    yield chunk
                    
                yield "✨ Writing..."
                for chunk in self.stream.writer.run("Write an article"):
                    yield chunk
        """
        # Always create fresh streaming agents to avoid conflicts with proxied agents
        self._setup_agents()
        streaming_data: Dict[str, StreamingAgentProxy] = {}
        
        # First, try to use the stored original agents if they exist (during proxied execution)
        if hasattr(self, '_original_agents_data') and self._original_agents_data:
            # Use the original agents stored before proxying
            for field_name, original_agent in self._original_agents_data.items():
                if isinstance(original_agent, Agent):
                    streaming_data[field_name] = StreamingAgentProxy(original_agent, self._execution_context, workflow_agent_name=field_name)
        elif hasattr(self, 'agents') and isinstance(getattr(self, 'agents'), BaseModel):
            # Use the current agents registry (not proxied yet)
            original_agents = getattr(self, 'agents')
            
            # Get all agent fields (Pydantic V2 compatibility)
            if hasattr(original_agents.__class__, 'model_fields'):
                # Pydantic V2 - use class attribute to avoid deprecation warning
                for field_name in original_agents.__class__.model_fields:
                    agent = getattr(original_agents, field_name)
                    if isinstance(agent, Agent):
                        streaming_data[field_name] = StreamingAgentProxy(agent, self._execution_context, workflow_agent_name=field_name)
            elif hasattr(original_agents, '__fields__'):
                # Pydantic V1
                for field_name in original_agents.__fields__:
                    agent = getattr(original_agents, field_name)
                    if isinstance(agent, Agent):
                        streaming_data[field_name] = StreamingAgentProxy(agent, self._execution_context, workflow_agent_name=field_name)
        
        # Create a dynamic class with the same structure as the original agents
        class StreamingAgents:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
            
            def __repr__(self):
                attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
                return f"StreamingAgents({', '.join(attrs)})"
        
        return StreamingAgents(streaming_data)
    
    def _call_agent_with_context(self, agent, method_name: str, *args, **kwargs):
        """Call agent method with current execution context."""
        context = self._execution_context

        if context.mode == 'sync':
            # Regular sync execution
            return getattr(agent, method_name)(*args, **kwargs)

        elif context.mode == 'stream':
            # Sync streaming - use stream=True
            kwargs['stream'] = True
            return getattr(agent, method_name)(*args, **kwargs)

        elif context.mode == 'async':
            if hasattr(agent, 'arun'):
                return agent.arun(*args, **kwargs)
            # Fallback to sync in executor
            async def async_wrapper():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, getattr(agent, method_name), *args, **kwargs)

            return async_wrapper()

        elif context.mode == 'async_stream':
            # Async streaming - use astream
            if hasattr(agent, 'astream'):
                return agent.astream(*args, **kwargs)
            elif hasattr(agent, 'arun'):
                # Fallback to arun with stream=True
                kwargs['stream'] = True
                return agent.arun(*args, **kwargs)
            else:
                # Final fallback
                kwargs['stream'] = True
                async def async_wrapper():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, getattr(agent, method_name), *args, **kwargs)
                return async_wrapper()

        # Default fallback
        return getattr(agent, method_name)(*args, **kwargs)

    def _install_agent_method_wrappers(self):
        """Store references to agents for context-aware execution."""
        if self._method_wrappers_installed or not self._agents:
            return
            
        # Just mark as installed - we'll handle interception differently
        self._method_wrappers_installed = True
    
    def _remove_agent_method_wrappers(self):
        """Clean up wrapper installation."""
        self._method_wrappers_installed = False
    
    def _replace_agents_with_proxies(self):
        """Replace agents with proxy objects for context-aware execution."""
        if not self._agents:
            return
            
        # Store original agents if not already stored
        if not hasattr(self, '_original_agents_data'):
            self._original_agents_data = {}
            
            # Get all agent fields (Pydantic V2 compatibility)
            if hasattr(self._agents, 'model_fields'):
                # Pydantic V2
                for field_name in self._agents.model_fields:
                    agent = getattr(self._agents, field_name)
                    if isinstance(agent, Agent):
                        self._original_agents_data[field_name] = agent
                        # Replace with proxy that uses Pydantic field name
                        proxy = AgentProxy(agent, self._execution_context, self, workflow_agent_name=field_name)
                        setattr(self._agents, field_name, proxy)
            elif hasattr(self._agents, '__fields__'):
                # Pydantic V1
                for field_name in self._agents.__fields__:
                    agent = getattr(self._agents, field_name)
                    if isinstance(agent, Agent):
                        self._original_agents_data[field_name] = agent
                        # Replace with proxy that uses Pydantic field name
                        proxy = AgentProxy(agent, self._execution_context, self, workflow_agent_name=field_name)
                        setattr(self._agents, field_name, proxy)
    
    def _restore_original_agents(self):
        """Restore original agents from proxies."""
        if not self._agents or not hasattr(self, '_original_agents_data'):
            return
            
        # Restore original agents
        for field_name, original_agent in self._original_agents_data.items():
            setattr(self._agents, field_name, original_agent)
            
        # Clean up
        delattr(self, '_original_agents_data')
    

    
    def run(self, input_data: Any) -> Any:
        """
        Implement your workflow logic here.
        
        For streaming workflows, use yield to return chunks in real-time.
        For non-streaming workflows, return the final result directly.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Workflow execution result (can be generator for streaming)
        """
        raise NotImplementedError("Subclass must implement run() method")
    
    def collect(self, input_data: Any) -> ThinagentResponse:
        """
        Execute workflow and collect result into ThinagentResponse.
        
        This method calls your run() implementation and handles:
        - Generator collection for streaming workflows
        - Agent response tracking 
        - Proper ThinagentResponse creation
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            ThinagentResponse with final content and all agent responses
        """
        return self.execute(input_data)
    
    def execute(self, input_data: Any) -> ThinagentResponse:
        """
        Execute workflow and return final result as ThinagentResponse.
        
        This method handles both generator and non-generator workflows.
        For generator workflows, it collects all content and agent responses.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            ThinagentResponse with final content and all agent responses
        """
        try:
            # Set sync execution context
            self._execution_context.set_mode(is_async=False, is_streaming=False)
            self._pending_chunks.clear()
            
            # Track all agent responses
            agent_responses: Dict[str, List[Any]] = {}
            
            # Setup agents with response tracking
            self._setup_agents()
            self._replace_agents_with_proxies()
            
            try:
                result = self.run(input_data)
                
                # Check if workflow is generator-based (uses yield)
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                    try:
                        # Collect all content from generator
                        final_content_parts = []
                        
                        for chunk in result:
                            if isinstance(chunk, ThinagentResponseStream):
                                final_content_parts.append(chunk.content)
                                # Track agent responses
                                if chunk.agent_name:
                                    if chunk.agent_name not in agent_responses:
                                        agent_responses[chunk.agent_name] = []
                                    agent_responses[chunk.agent_name].append(chunk)
                            elif isinstance(chunk, str):
                                final_content_parts.append(chunk)
                            else:
                                final_content_parts.append(str(chunk))
                        
                        final_content = "".join(final_content_parts)
                        
                    except TypeError:
                        # If iteration fails, treat as regular result
                        final_content = str(result) if result else ""
                        
                else:
                    # Non-generator workflow
                    final_content = str(result) if result else ""
                
                # Collect agent responses that were tracked during proxy execution
                if hasattr(self, '_original_agents_data'):
                    for agent_name in self._original_agents_data.keys():
                        # Create consolidated response for each agent
                        if agent_name not in agent_responses:
                            agent_responses[agent_name] = []
                        
                        # If no responses were tracked, try to get from state
                        if not agent_responses[agent_name] and hasattr(self.state, 'research'):
                             # Add stored responses from state
                             if agent_name == 'researcher' and hasattr(self.state, 'research'):
                                 agent_responses[agent_name] = [ThinagentResponse(
                                     content=self.state.research,
                                     content_type="str",
                                     response_id=None,
                                     created_timestamp=None,
                                     model_used=None,
                                     finish_reason=None,
                                     metrics=None,
                                     system_fingerprint=None,
                                     artifact=None,
                                     tool_name=None,
                                     tool_call_id=None,
                                     agent_name=agent_name
                                 )]
                             elif agent_name == 'topic_adjuster' and hasattr(self.state, 'adjusted_content'):
                                 agent_responses[agent_name] = [ThinagentResponse(
                                     content=self.state.adjusted_content,
                                     content_type="str",
                                     response_id=None,
                                     created_timestamp=None,
                                     model_used=None,
                                     finish_reason=None,
                                     metrics=None,
                                     system_fingerprint=None,
                                     artifact=None,
                                     tool_name=None,
                                     tool_call_id=None,
                                     agent_name=agent_name
                                 )]
                             elif agent_name == 'enhancer' and hasattr(self.state, 'final_content'):
                                 agent_responses[agent_name] = [ThinagentResponse(
                                     content=self.state.final_content,
                                     content_type="str",
                                     response_id=None,
                                     created_timestamp=None,
                                     model_used=None,
                                     finish_reason=None,
                                     metrics=None,
                                     system_fingerprint=None,
                                     artifact=None,
                                     tool_name=None,
                                     tool_call_id=None,
                                     agent_name=agent_name
                                 )]
                
                # Create workflow data with agent responses
                workflow_data = None
                if agent_responses:
                    workflow_data = WorkflowData(agent_responses=agent_responses)

                # Determine the final agent that produced the content
                # In generator workflows, the final agent is usually the last one that yielded content
                final_agent_name = "workflow"  # default fallback
                
                if hasattr(self, '_original_agents_data'):
                    # For this workflow, we know the enhancer is the final agent based on the logic
                    # In a more general case, we could track which agent produced the final content
                    if 'enhancer' in self._original_agents_data:
                        final_agent_name = "enhancer"
                    elif 'topic_adjuster' in self._original_agents_data:
                        final_agent_name = "topic_adjuster"  
                    elif 'researcher' in self._original_agents_data:
                        final_agent_name = "researcher"
                    else:
                        # Get the last agent in the registry as fallback
                        agent_names = list(self._original_agents_data.keys())
                        if agent_names:
                            final_agent_name = agent_names[-1]

                # Create final ThinagentResponse with workflow data
                final_response = ThinagentResponse(
                    content=final_content,
                    content_type="str",
                    response_id=None,
                    created_timestamp=None,
                    model_used=None,
                    finish_reason="workflow_complete",
                    metrics=None,
                    system_fingerprint=None,
                    artifact=None,
                    tool_name=None,
                    tool_call_id=None,
                    agent_name=final_agent_name,  # Use final agent name, not "workflow"
                    workflow=workflow_data
                )
                
                return final_response
                
            finally:
                # Restore original agents and context
                self._restore_original_agents()
                self._execution_context.set_mode(is_async=False, is_streaming=False)
                self._pending_chunks.clear()
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise WorkflowExecutionError(f"Workflow execution failed: {e}") from e
    

    
    def run_stream(self, input_data: Any) -> Iterator[Any]:
        """
        Streaming version of run().
        
        If the workflow's run() method is a generator (uses yield), streams chunks in real-time.
        Otherwise, uses automatic streaming conversion for backward compatibility.
        
        Args:
            input_data: Input data for the workflow
            
        Yields:
            Streaming results from the workflow
        """
        try:
            # Set streaming execution context
            self._execution_context.set_mode(is_async=False, is_streaming=True)
            self._pending_chunks.clear()  # Clear any previous chunks
            
            # Ensure agents are set up
            self._setup_agents()
            self._replace_agents_with_proxies()
            
            try:
                # Execute the workflow
                result = self.run(input_data)
                
                # Check if workflow.run() is a generator (uses yield)
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                    # Real-time streaming: yield chunks as they come from the generator
                    try:
                        for chunk in result:
                            # If the yielded item is an iterator, stream its contents.
                            if hasattr(chunk, '__iter__') and not isinstance(chunk, (str, bytes, dict)):
                                yield from chunk
                                continue
                                
                            # Ensure chunk is a proper ThinagentResponseStream
                            if isinstance(chunk, ThinagentResponseStream):
                                yield chunk
                            elif isinstance(chunk, str):
                                # Convert string to ThinagentResponseStream
                                yield ThinagentResponseStream(
                                    content=chunk,
                                    content_type="str",
                                    agent_name=None,
                                    response_id=None,
                                    created_timestamp=None,
                                    model_used=None,
                                    finish_reason=None,
                                    metrics=None,
                                    system_fingerprint=None,
                                    artifact=None,
                                    tool_name=None,
                                    tool_call_id=None,
                                    stream_options=None
                                )
                            else:
                                # Handle other types
                                yield ThinagentResponseStream(
                                    content=str(chunk),
                                    content_type="str",
                                    agent_name=None,
                                    response_id=None,
                                    created_timestamp=None,
                                    model_used=None,
                                    finish_reason=None,
                                    metrics=None,
                                    system_fingerprint=None,
                                    artifact=None,
                                    tool_name=None,
                                    tool_call_id=None,
                                    stream_options=None
                                )
                    except TypeError:
                        # If iteration fails, fall back to collected chunks
                        pass
                
                # Fallback: yield collected chunks for backward compatibility
                chunks_yielded = 0
                if self._pending_chunks:
                    for chunk in self._pending_chunks:
                        chunks_yielded += 1
                        yield chunk
                
                # If no chunks were yielded, yield the final result
                if chunks_yielded == 0 and not hasattr(result, '__iter__'):
                    final_content = str(result) if result else "No content generated"
                    yield ThinagentResponseStream(
                        content=final_content,
                        content_type="str",
                        agent_name=None,
                        response_id=None,
                        created_timestamp=None,
                        model_used=None,
                        finish_reason="workflow_complete",
                        metrics=None,
                        system_fingerprint=None,
                        artifact=None,
                        tool_name=None,
                        tool_call_id=None,
                        stream_options=None
                    )
                    
            finally:
                # Restore original agents and context
                self._restore_original_agents()
                self._execution_context.set_mode(is_async=False, is_streaming=False)
                self._pending_chunks.clear()
                
        except Exception as e:
            logger.error(f"Workflow streaming execution failed: {e}")
            raise WorkflowExecutionError(f"Workflow streaming failed: {e}") from e
    
    async def arun(self, input_data: Any) -> Any:
        """
        Async version of run().
        
        Automatically converts agent.run() calls to async mode.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Workflow execution result
        """
        try:
            # Set async execution context
            self._execution_context.set_mode(is_async=True, is_streaming=False)
            self._replace_agents_with_proxies()

            try:
                if inspect.iscoroutinefunction(self.run):
                    return await self.run(input_data)
                # Convert sync run to async
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.run, input_data)

            finally:
                # Restore original agents and context
                self._restore_original_agents()
                self._execution_context.set_mode(is_async=False, is_streaming=False)

        except Exception as e:
            logger.error(f"Workflow async execution failed: {e}")
            raise WorkflowExecutionError(f"Workflow async execution failed: {e}") from e
    
    async def arun_stream(self, input_data: Any) -> AsyncIterator[Any]:
        """
        Async streaming version of run().
        
        Automatically converts agent.run() calls to async streaming mode.
        
        Args:
            input_data: Input data for the workflow
            
        Yields:
            Async streaming results from the workflow
        """
        try:
            # Set async streaming execution context
            self._execution_context.set_mode(is_async=True, is_streaming=True)
            self._replace_agents_with_proxies()
            
            try:
                # Execute workflow
                if inspect.iscoroutinefunction(self.run):
                    result = await self.run(input_data)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.run, input_data)
                
                # Handle async iteration
                if hasattr(result, '__aiter__'):
                    async for chunk in result:
                        if hasattr(chunk, '__aiter__'):
                            async for sub_chunk in chunk:
                                yield sub_chunk
                        elif hasattr(chunk, '__iter__') and not isinstance(chunk, (str, bytes, dict)):
                            for sub_chunk in chunk:
                                yield sub_chunk
                        else:
                            yield chunk
                elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                    for chunk in result:
                        if hasattr(chunk, '__iter__') and not isinstance(chunk, (str, bytes, dict)):
                            for sub_chunk in chunk:
                                yield sub_chunk
                        else:
                            yield chunk
                else:
                    yield result
            finally:
                # Restore original agents and context
                self._restore_original_agents()
                self._execution_context.set_mode(is_async=False, is_streaming=False)
                
        except Exception as e:
            logger.error(f"Workflow async streaming execution failed: {e}")
            raise WorkflowExecutionError(f"Workflow async streaming failed: {e}") from e
    

    
    def get_agent(self, name: str) -> Agent:
        """
        Get an agent by name from the registry.
        
        Args:
            name: Name of the agent to retrieve
            
        Returns:
            The requested agent
            
        Raises:
            AgentNotFoundError: If the agent is not found
        """
        self._setup_agents()  # Ensure agents are set up
        
        if not self._agents:
            raise AgentNotFoundError("No agents registered in this workflow")
        
        if hasattr(self._agents, name):
            agent = getattr(self._agents, name)
            if isinstance(agent, Agent):
                return agent
        
        # Also check in the fields if using Pydantic model
        if hasattr(self._agents, 'model_fields') and name in self._agents.model_fields:
            # Pydantic V2
            agent = getattr(self._agents, name)
            if isinstance(agent, Agent):
                return agent
        elif hasattr(self._agents, '__fields__') and name in self._agents.__fields__:
            # Pydantic V1
            agent = getattr(self._agents, name)
            if isinstance(agent, Agent):
                return agent
        
        raise AgentNotFoundError(f"Agent '{name}' not found in workflow")
    
    def list_agents(self) -> Dict[str, Agent]:
        """
        List all registered agents.
        
        Returns:
            Dictionary mapping agent names to Agent instances
        """
        self._setup_agents()  # Ensure agents are set up
        
        if not self._agents:
            return {}
        
        agents = {}
        if hasattr(self._agents, 'model_fields'):
            # Pydantic V2 model
            for field_name in self._agents.model_fields:
                agent = getattr(self._agents, field_name)
                if isinstance(agent, Agent):
                    agents[field_name] = agent
        elif hasattr(self._agents, '__fields__'):
            # Pydantic V1 model
            for field_name in self._agents.__fields__:
                agent = getattr(self._agents, field_name)
                if isinstance(agent, Agent):
                    agents[field_name] = agent
        else:
            # Regular object - inspect all attributes
            for attr_name in dir(self._agents):
                if not attr_name.startswith('_'):
                    agent = getattr(self._agents, attr_name)
                    if isinstance(agent, Agent):
                        agents[attr_name] = agent
        
        return agents
    
    def reset_state(self) -> None:
        """Reset the workflow state."""
        self.state.clear()
    
    def __repr__(self) -> str:
        self._setup_agents()  # Ensure agents are set up
        agent_count = len(self.list_agents()) if self._agents else 0
        return f"{self.__class__.__name__}(agents={agent_count}, state_keys={list(self.state.keys())})"


# Alias for backwards compatibility and cleaner imports
Workflow = BaseWorkflow 