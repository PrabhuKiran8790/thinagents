"""
Memory module for ThinAgents providing conversation history storage and retrieval.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ConversationInfo(TypedDict):
    """Type definition for conversation metadata."""
    conversation_id: str
    message_count: int
    last_message: Optional[Dict[str, Any]]
    created_at: Optional[str]
    updated_at: Optional[str]


class BaseMemory(ABC):
    """
    Abstract base class for memory implementations.
    
    This class defines the interface that all memory backends must implement.
    Memory stores conversation history as a list of message dictionaries.
    """
    
    @abstractmethod
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a given conversation ID.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            List of message dictionaries in chronological order
        """
        pass
    
    @abstractmethod
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Store a new message in the conversation history.
        
        Args:
            conversation_id: Unique identifier for the conversation
            message: Message dictionary to store
        """
        pass
    
    @abstractmethod
    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear all messages for a specific conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
        """
        pass
    
    @abstractmethod
    def list_conversation_ids(self) -> List[str]:
        """
        List all conversation IDs in the memory store.
        
        Returns:
            List of conversation IDs as strings
        """
        pass
    
    def list_conversations(self) -> List[ConversationInfo]:
        """
        List all conversations with detailed metadata.
        
        Returns:
            List of conversation info dictionaries with metadata
        """
        conversation_infos: List[ConversationInfo] = []
        
        for conversation_id in self.list_conversation_ids():
            messages = self.get_messages(conversation_id)
            
            # Get first and last message timestamps for created_at/updated_at
            created_at = None
            updated_at = None
            last_message = None
            
            if messages:
                last_message = messages[-1]
                
                # Get timestamps from first and last messages
                if "timestamp" in messages[0]:
                    created_at = messages[0]["timestamp"]
                if "timestamp" in last_message:
                    updated_at = last_message["timestamp"]
            
            conversation_info: ConversationInfo = {
                "conversation_id": conversation_id,
                "message_count": len(messages),
                "last_message": last_message,
                "created_at": created_at,
                "updated_at": updated_at,
            }
            
            conversation_infos.append(conversation_info)
        
        # Sort by updated_at (most recent first), fallback to conversation_id
        conversation_infos.sort(
            key=lambda x: (x["updated_at"] or "", x["conversation_id"]), 
            reverse=True
        )
        
        return conversation_infos
    
    def add_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Add multiple messages to a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store
        """
        for message in messages:
            self.add_message(conversation_id, message)
    
    def get_conversation_length(self, conversation_id: str) -> int:
        """
        Get the number of messages in a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Number of messages in the conversation
        """
        return len(self.get_messages(conversation_id))
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """
        Check if a conversation exists.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            True if conversation exists, False otherwise
        """
        return conversation_id in self.list_conversation_ids()
    
    def get_conversation_info(self, conversation_id: str) -> Optional[ConversationInfo]:
        """
        Get detailed information about a specific conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            ConversationInfo dictionary or None if conversation doesn't exist
        """
        if not self.conversation_exists(conversation_id):
            return None
            
        messages = self.get_messages(conversation_id)
        
        created_at = None
        updated_at = None
        last_message = None
        
        if messages:
            last_message = messages[-1]
            
            if "timestamp" in messages[0]:
                created_at = messages[0]["timestamp"]
            if "timestamp" in last_message:
                updated_at = last_message["timestamp"]
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "last_message": last_message,
            "created_at": created_at,
            "updated_at": updated_at,
        }


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


class FileMemory(BaseMemory):
    """
    File-based implementation of memory storage.
    
    This implementation stores each conversation as a separate JSON file
    in the specified directory. Provides persistence across application restarts.
    """
    
    def __init__(self, storage_dir: str = "./conversations"):
        """
        Initialize the file-based memory store.
        
        Args:
            storage_dir: Directory to store conversation files
        """
        import os
        self.storage_dir = storage_dir
        
        # Create directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        logger.debug(f"Initialized FileMemory with storage_dir: {storage_dir}")
    
    def _get_file_path(self, conversation_id: str) -> str:
        """Get the file path for a conversation."""
        import os
        # Sanitize conversation_id for filesystem
        safe_id = "".join(c for c in conversation_id if c.isalnum() or c in ('-', '_', '.'))
        return os.path.join(self.storage_dir, f"{safe_id}.json")
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages from file."""
        import os
        file_path = self._get_file_path(conversation_id)
        
        if not os.path.exists(file_path):
            logger.debug(f"No file found for conversation '{conversation_id}'")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            logger.debug(f"Retrieved {len(messages)} messages for conversation '{conversation_id}' from file")
            return messages
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading conversation file '{file_path}': {e}")
            return []
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Add a message to file."""
        messages = self.get_messages(conversation_id)
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message = message.copy()
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        messages.append(message)
        
        file_path = self._get_file_path(conversation_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            logger.debug(f"Added message to conversation '{conversation_id}' in file (total: {len(messages)})")
        except IOError as e:
            logger.error(f"Error writing to conversation file '{file_path}': {e}")
            raise
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation file."""
        import os
        file_path = self._get_file_path(conversation_id)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleared conversation '{conversation_id}' (deleted file)")
            except OSError as e:
                logger.error(f"Error deleting conversation file '{file_path}': {e}")
                raise
        else:
            logger.warning(f"Conversation file '{file_path}' not found for clearing")
    
    def list_conversation_ids(self) -> List[str]:
        """List all conversation IDs by scanning files."""
        import os
        conversations: List[str] = []

        if not os.path.exists(self.storage_dir):
            return conversations

        conversations.extend(
            filename[:-5]
            for filename in os.listdir(self.storage_dir)
            if filename.endswith('.json')
        )
        return conversations 