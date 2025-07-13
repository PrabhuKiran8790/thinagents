"""
In-memory storage for ThinAgents conversation history.
"""

import asyncio
import logging
from typing import Any, Dict, List
from datetime import datetime, timezone

from thinagents.memory.base_memory import BaseMemory, ConversationInfo

logger = logging.getLogger(__name__)


class InMemoryStore(BaseMemory):
    """
    In-memory implementation of memory storage.
    
    This implementation stores conversations in memory and will be lost
    when the application terminates. Useful for development and testing.
    Tool artifacts are stored directly in tool messages when enabled.
    Supports both sync and async operations.
    """
    
    def __init__(self, store_tool_artifacts: bool = False):
        """
        Initialize the in-memory store.
        
        Args:
            store_tool_artifacts: If True, include tool artifacts in tool messages.
                Defaults to False to avoid unnecessary memory usage.
        """
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.store_tool_artifacts = store_tool_artifacts
        self._lock = asyncio.Lock()  # For thread safety in async operations
        logger.debug(f"Initialized InMemoryStore with store_tool_artifacts={store_tool_artifacts}")
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages from memory."""
        messages = self._conversations.get(conversation_id, [])
        logger.debug(f"Retrieved {len(messages)} messages for conversation '{conversation_id}'")
        return messages.copy()  # Return a copy to prevent external modification
    
    async def aget_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Async version of get_messages."""
        async with self._lock:
            messages = self._conversations.get(conversation_id, [])
            logger.debug(f"Retrieved {len(messages)} messages for conversation '{conversation_id}' (async)")
            return messages.copy()
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Add a message to memory."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message = message.copy()
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        self._conversations[conversation_id].append(message)
        logger.debug(f"Added message to conversation '{conversation_id}' (total: {len(self._conversations[conversation_id])})")
    
    async def aadd_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Async version of add_message."""
        async with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            
            # Add timestamp if not present
            if "timestamp" not in message:
                message = message.copy()
                message["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            self._conversations[conversation_id].append(message)
            logger.debug(f"Added message to conversation '{conversation_id}' (total: {len(self._conversations[conversation_id])}) (async)")
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation from memory."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Cleared conversation '{conversation_id}'")
        else:
            logger.warning(f"Conversation '{conversation_id}' not found for clearing")
    
    async def aclear_conversation(self, conversation_id: str) -> None:
        """Async version of clear_conversation."""
        async with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                logger.info(f"Cleared conversation '{conversation_id}' (async)")
            else:
                logger.warning(f"Conversation '{conversation_id}' not found for clearing")
    
    def list_conversation_ids(self) -> List[str]:
        """List all conversation IDs."""
        return list(self._conversations.keys())
    
    async def alist_conversation_ids(self) -> List[str]:
        """Async version of list_conversation_ids."""
        async with self._lock:
            return list(self._conversations.keys())
    
    def list_conversations(self) -> List[ConversationInfo]:
        """List all conversations with detailed metadata."""
        conversations = []
        for conversation_id in self._conversations.keys():
            messages = self._conversations[conversation_id]
            
            created_at = None
            updated_at = None
            last_message = None
            
            if messages:
                last_message = messages[-1]
                
                if "timestamp" in messages[0]:
                    created_at = messages[0]["timestamp"]
                if "timestamp" in last_message:
                    updated_at = last_message["timestamp"]
            
            conversations.append({
                "conversation_id": conversation_id,
                "message_count": len(messages),
                "last_message": last_message,
                "created_at": created_at,
                "updated_at": updated_at,
            })
        
        return conversations
    
    async def alist_conversations(self) -> List[ConversationInfo]:
        """Async version of list_conversations."""
        async with self._lock:
            conversations = []
            for conversation_id in self._conversations.keys():
                messages = self._conversations[conversation_id]
                
                created_at = None
                updated_at = None
                last_message = None
                
                if messages:
                    last_message = messages[-1]
                    
                    if "timestamp" in messages[0]:
                        created_at = messages[0]["timestamp"]
                    if "timestamp" in last_message:
                        updated_at = last_message["timestamp"]
                
                conversations.append({
                    "conversation_id": conversation_id,
                    "message_count": len(messages),
                    "last_message": last_message,
                    "created_at": created_at,
                    "updated_at": updated_at,
                })
            
            return conversations
    
    def clear_all(self) -> None:
        """Clear all conversations from memory."""
        count = len(self._conversations)
        self._conversations.clear()
        logger.info(f"Cleared all conversations ({count} total)")
    
    async def aclear_all(self) -> None:
        """Async version of clear_all."""
        async with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            logger.info(f"Cleared all conversations ({count} total) (async)")
    
    # Optimized batch operations
    async def aadd_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Optimized async version of add_messages that adds all messages at once.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store
        """
        if not messages:
            return
        
        async with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            
            # Add timestamps to messages that don't have them
            processed_messages = []
            for message in messages:
                if "timestamp" not in message:
                    message = message.copy()
                    message["timestamp"] = datetime.now(timezone.utc).isoformat()
                processed_messages.append(message)
            
            self._conversations[conversation_id].extend(processed_messages)
            logger.debug(f"Added {len(processed_messages)} messages to conversation '{conversation_id}' (total: {len(self._conversations[conversation_id])}) (async batch)")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_messages = sum(len(messages) for messages in self._conversations.values())
        return {
            "total_conversations": len(self._conversations),
            "total_messages": total_messages,
            "conversations": {
                conv_id: len(messages) 
                for conv_id, messages in self._conversations.items()
            }
        }
    
    async def aget_memory_usage(self) -> Dict[str, Any]:
        """Async version of get_memory_usage."""
        async with self._lock:
            total_messages = sum(len(messages) for messages in self._conversations.values())
            return {
                "total_conversations": len(self._conversations),
                "total_messages": total_messages,
                "conversations": {
                    conv_id: len(messages) 
                    for conv_id, messages in self._conversations.items()
                }
            } 