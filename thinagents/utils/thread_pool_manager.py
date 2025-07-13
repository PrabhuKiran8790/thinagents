"""
Thread pool manager for efficient tool execution concurrency.

This module provides a global thread pool manager that can be shared across
multiple agent instances to optimize resource usage and provide better
concurrency control for tool execution.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)



@dataclass
class ThreadPoolConfig:
    """Configuration for thread pool management."""
    max_workers: int = 10
    thread_name_prefix: str = "ThinAgentsTool"
    
    def __post_init__(self):
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


class ThreadPoolManager:
    """
    Global thread pool manager for efficient tool execution.
    
    This manager provides a singleton thread pool that can be shared across
    multiple agent instances to optimize resource usage and provide better
    concurrency control.
    """
    
    _instance: Optional['ThreadPoolManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[ThreadPoolConfig] = None) -> 'ThreadPoolManager':
        """Singleton implementation with lazy initialization."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, config: Optional[ThreadPoolConfig] = None):
        """Initialize the thread pool manager."""
        if getattr(self, '_initialized', False):
            return
            
        self._config = config or ThreadPoolConfig()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._shutdown_event = threading.Event()
        self._executor_lock = threading.Lock()
        self._initialized = True
        
        logger.info(f"ThreadPoolManager initialized with max_workers={self._config.max_workers}")
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get the thread pool executor, creating it if necessary."""
        if self._executor is None or self._executor._shutdown:
            with self._executor_lock:
                if self._executor is None or self._executor._shutdown:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._config.max_workers,
                        thread_name_prefix=self._config.thread_name_prefix
                    )
                    logger.debug(f"Created new ThreadPoolExecutor with {self._config.max_workers} workers")
        return self._executor
    

    
    def submit_tool_execution(
        self,
        tool_func: Callable,
        tool_args: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Future:
        """
        Submit a tool execution to the thread pool.
        
        Args:
            tool_func: The tool function to execute
            tool_args: Arguments to pass to the tool
            timeout: Optional timeout for the execution
            
        Returns:
            Future object representing the execution
        """
        return self.executor.submit(tool_func, **tool_args)
    
    def execute_tools_concurrently(
        self,
        tool_calls: List[tuple],  # List of (tool_func, tool_args) tuples
        timeout: Optional[float] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Union[Any, Exception]]:
        """
        Execute multiple tools concurrently.
        
        Args:
            tool_calls: List of (tool_func, tool_args) tuples
            timeout: Optional timeout for each tool execution
            max_concurrent: Maximum number of concurrent executions (defaults to config max_workers)
            
        Returns:
            List of results in the same order as tool_calls
        """
        if not tool_calls:
            return []

        # Submit all tasks
        future_to_index = {}
        for i, (tool_func, tool_args) in enumerate(tool_calls):
            future = self.submit_tool_execution(tool_func, tool_args, timeout)
            future_to_index[future] = i

        # Collect results
        results: List[Union[Any, Exception]] = [None] * len(tool_calls)
        exceptions = {}

        try:
            for future in as_completed(future_to_index.keys(), timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result(timeout=0.1)  # Small timeout since future is done
                except Exception as e:
                    exceptions[index] = e
                    logger.error(f"Tool execution {index} failed: {e}")
        except Exception as e:
            logger.error(f"Error in concurrent tool execution: {e}")
            # Cancel remaining futures
            for future in future_to_index:
                if not future.done():
                    future.cancel()
            raise

        # Raise exceptions if any occurred
        if exceptions:
            # For now, we'll include exceptions in the results
            # This allows the caller to handle them appropriately
            for index, exception in exceptions.items():
                results[index] = exception

        return results
    
    @contextmanager
    def execution_context(self, max_concurrent: Optional[int] = None):
        """
        Context manager for controlling concurrent executions.
        
        Args:
            max_concurrent: Maximum number of concurrent executions in this context
        """
        original_max_workers = self._config.max_workers
        try:
            if max_concurrent is not None:
                self._config.max_workers = max_concurrent
                # If we need to restart the executor with new settings
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                    self._executor = None
            yield self
        finally:
            self._config.max_workers = original_max_workers
            if max_concurrent is not None and self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool manager.
        
        Args:
            wait: Whether to wait for running tasks to complete
        """
        if self._executor is not None:
            logger.info("Shutting down ThreadPoolManager")
            self._executor.shutdown(wait=wait)
            self._executor = None
        
        self._shutdown_event.set()
    

    
    @classmethod
    def get_instance(cls, config: Optional[ThreadPoolConfig] = None) -> 'ThreadPoolManager':
        """Get the singleton instance of ThreadPoolManager."""
        return cls(config)
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        if cls._instance is not None:
            cls._instance.shutdown(wait=True)
        cls._instance = None
    
    def __del__(self):
        """Cleanup when the manager is destroyed."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)


# Global instance accessor
def get_thread_pool_manager(config: Optional[ThreadPoolConfig] = None) -> ThreadPoolManager:
    """Get the global thread pool manager instance."""
    return ThreadPoolManager.get_instance(config)


# Async utilities for integration with asyncio
async def execute_tool_in_thread(
    tool_func: Callable,
    tool_args: Dict[str, Any],
    timeout: Optional[float] = None,
    thread_pool_manager: Optional[ThreadPoolManager] = None
) -> Any:
    """
    Execute a tool function in a thread pool using asyncio.
    
    Args:
        tool_func: The tool function to execute
        tool_args: Arguments to pass to the tool
        timeout: Optional timeout for the execution
        thread_pool_manager: Optional thread pool manager instance
        
    Returns:
        The result of the tool execution
    """
    manager = thread_pool_manager or get_thread_pool_manager()

    # Submit to thread pool
    future = manager.submit_tool_execution(tool_func, tool_args, timeout)

    # Wait for completion using asyncio
    try:
        return await asyncio.wrap_future(future)
    except asyncio.TimeoutError:
        future.cancel()
        raise 