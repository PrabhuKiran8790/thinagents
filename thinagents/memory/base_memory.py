"""
Base memory module for ThinAgents providing conversation history storage and retrieval.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

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
    Provides both sync and async methods for flexibility.
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
    async def aget_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Async version of get_messages.
        
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
    async def aadd_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Async version of add_message.
        
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
    async def aclear_conversation(self, conversation_id: str) -> None:
        """
        Async version of clear_conversation.
        
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
    
    @abstractmethod
    async def alist_conversation_ids(self) -> List[str]:
        """
        Async version of list_conversation_ids.
        
        Returns:
            List of conversation IDs as strings
        """
        pass
    
    @abstractmethod
    def list_conversations(self) -> List[ConversationInfo]:
        """
        List all conversations with detailed metadata.
        
        This method should be implemented efficiently by subclasses,
        avoiding loading full message histories for each conversation.

        Returns:
            List of conversation info dictionaries with metadata
        """
        pass
    
    @abstractmethod
    async def alist_conversations(self) -> List[ConversationInfo]:
        """
        Async version of list_conversations.
        
        This method should be implemented efficiently by subclasses,
        avoiding loading full message histories for each conversation.

        Returns:
            List of conversation info dictionaries with metadata
        """
        pass
    
    def add_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Add multiple messages to a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store
        """
        for message in messages:
            self.add_message(conversation_id, message)
    
    async def aadd_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Async version of add_messages.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store
        """
        for message in messages:
            await self.aadd_message(conversation_id, message)
    
    def get_conversation_length(self, conversation_id: str) -> int:
        """
        Get the number of messages in a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Number of messages in the conversation
        """
        return len(self.get_messages(conversation_id))
    
    async def aget_conversation_length(self, conversation_id: str) -> int:
        """
        Async version of get_conversation_length.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Number of messages in the conversation
        """
        return len(await self.aget_messages(conversation_id))
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """
        Check if a conversation exists.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            True if conversation exists, False otherwise
        """
        return conversation_id in self.list_conversation_ids()
    
    async def aconversation_exists(self, conversation_id: str) -> bool:
        """
        Async version of conversation_exists.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            True if conversation exists, False otherwise
        """
        return conversation_id in await self.alist_conversation_ids()
    
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
    
    async def aget_conversation_info(self, conversation_id: str) -> Optional[ConversationInfo]:
        """
        Async version of get_conversation_info.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            ConversationInfo dictionary or None if conversation doesn't exist
        """
        if not await self.aconversation_exists(conversation_id):
            return None
            
        messages = await self.aget_messages(conversation_id)
        
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