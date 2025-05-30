"""
In-memory storage for ThinAgents conversation history.
"""

import logging
from typing import Any, Dict, List
from datetime import datetime, timezone

from thinagents.memory.base_memory import BaseMemory # Adjusted import

logger = logging.getLogger(__name__)


class InMemoryStore(BaseMemory):
    """
    In-memory implementation of memory storage.
    
    This implementation stores conversations in memory and will be lost
    when the application terminates. Useful for development and testing.
    Tool artifacts are stored directly in tool messages when enabled.
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
        logger.debug(f"Initialized InMemoryStore with store_tool_artifacts={store_tool_artifacts}")
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages from memory."""
        messages = self._conversations.get(conversation_id, [])
        logger.debug(f"Retrieved {len(messages)} messages for conversation '{conversation_id}'")
        return messages.copy()  # Return a copy to prevent external modification
    
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
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation from memory."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Cleared conversation '{conversation_id}'")
        else:
            logger.warning(f"Conversation '{conversation_id}' not found for clearing")
    
    def list_conversation_ids(self) -> List[str]:
        """List all conversation IDs."""
        return list(self._conversations.keys())
    
    def clear_all(self) -> None:
        """Clear all conversations from memory."""
        count = len(self._conversations)
        self._conversations.clear()
        logger.info(f"Cleared all conversations ({count} total)") 