"""
File-based storage for ThinAgents conversation history.
"""

import json
import logging
import os
from typing import Any, Dict, List
from datetime import datetime, timezone

from thinagents.memory.base_memory import BaseMemory # Adjusted import

logger = logging.getLogger(__name__)


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
        self.storage_dir = storage_dir
        
        # Create directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        logger.debug(f"Initialized FileMemory with storage_dir: {storage_dir}")
    
    def _get_file_path(self, conversation_id: str) -> str:
        """Get the file path for a conversation."""
        # Sanitize conversation_id for filesystem
        safe_id = "".join(c for c in conversation_id if c.isalnum() or c in ('-', '_', '.'))
        return os.path.join(self.storage_dir, f"{safe_id}.json")
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages from file."""
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
        conversations: List[str] = []

        if not os.path.exists(self.storage_dir):
            return conversations

        conversations.extend(
            filename[:-5]
            for filename in os.listdir(self.storage_dir)
            if filename.endswith('.json')
        )
        return conversations 