"""
File-based storage for ThinAgents conversation history.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone

from thinagents.memory.base_memory import BaseMemory, ConversationInfo

logger = logging.getLogger(__name__)

# Check if aiofiles is available
try:
    import aiofiles  # type: ignore
    import aiofiles.os  # type: ignore
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False


class FileMemory(BaseMemory):
    """
    File-based implementation of memory storage.
    
    This implementation stores each conversation as a separate file.
    It can use either JSONL (default, efficient for writes) or JSON format.
    Provides persistence across application restarts.
    Supports both sync and async operations for optimal performance.
    """
    
    def __init__(self, storage_dir: str = "./conversations", file_format: Literal["jsonl", "json"] = "jsonl"):
        """
        Initialize the file-based memory store.
        
        Args:
            storage_dir: Directory to store conversation files.
            file_format: The file format to use. Either 'jsonl' (default) or 'json'.
        """
        self.storage_dir = storage_dir
        if file_format not in ["jsonl", "json"]:
            raise ValueError("file_format must be either 'jsonl' or 'json'")
        self.file_format = file_format
        
        # Create directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        logger.debug(f"Initialized FileMemory with storage_dir: {storage_dir} and format: {file_format}")
        
        # Warn if aiofiles is not available
        if not AIOFILES_AVAILABLE:
            logger.warning("aiofiles not available. Async operations will use sync methods in thread pool.")
    
    def _get_file_path(self, conversation_id: str) -> str:
        """Get the file path for a conversation."""
        # Sanitize conversation_id for filesystem
        safe_id = "".join(c for c in conversation_id if c.isalnum() or c in ('-', '_', '.'))
        return os.path.join(self.storage_dir, f"{safe_id}.{self.file_format}")
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages from a file."""
        file_path = self._get_file_path(conversation_id)
        
        if not os.path.exists(file_path):
            logger.debug(f"No file found for conversation '{conversation_id}'")
            return []
        
        messages: List[Dict[str, Any]] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if self.file_format == "jsonl":
                    messages.extend(json.loads(line) for line in f if line.strip())
                else:
                    content = f.read()
                    if content:
                        messages = json.loads(content)
            logger.debug(f"Retrieved {len(messages)} messages for conversation '{conversation_id}' from file")
            return messages
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading conversation file '{file_path}': {e}")
            return []
    
    async def aget_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Async version of get_messages."""
        if not AIOFILES_AVAILABLE:
            # Fallback to sync version in thread pool
            return await asyncio.to_thread(self.get_messages, conversation_id)
        
        file_path = self._get_file_path(conversation_id)
        
        if not await aiofiles.os.path.exists(file_path):
            logger.debug(f"No file found for conversation '{conversation_id}'")
            return []
        
        messages: List[Dict[str, Any]] = []
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                if self.file_format == "jsonl":
                    async for line in f:
                        if line.strip():
                            messages.append(json.loads(line))
                else:
                    content = await f.read()
                    if content:
                        messages = json.loads(content)

            logger.debug(f"Retrieved {len(messages)} messages for conversation '{conversation_id}' from file (async)")
            return messages
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading conversation file '{file_path}': {e}")
            return []
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Add a message to file."""
        if "timestamp" not in message:
            message = message.copy()
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        file_path = self._get_file_path(conversation_id)

        try:
            if self.file_format == "jsonl":
                line_to_write = json.dumps(message, ensure_ascii=False) + "\n"
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(line_to_write)
                logger.debug(f"Appended message to conversation '{conversation_id}' file")
            else: # json
                messages = self.get_messages(conversation_id)
                messages.append(message)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, indent=2, ensure_ascii=False)
                logger.debug(f"Added message to conversation '{conversation_id}' file (total: {len(messages)})")
        except IOError as e:
            logger.error(f"Error writing to conversation file '{file_path}': {e}")
            raise
    
    async def aadd_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Async version of add_message."""
        if not AIOFILES_AVAILABLE:
            # Fallback to sync version in thread pool
            await asyncio.to_thread(self.add_message, conversation_id, message)
            return
        
        if "timestamp" not in message:
            message = message.copy()
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        file_path = self._get_file_path(conversation_id)
        try:
            if self.file_format == "jsonl":
                line_to_write = json.dumps(message, ensure_ascii=False) + "\n"
                async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                    await f.write(line_to_write)
                logger.debug(f"Appended message to conversation '{conversation_id}' file (async)")
            else: # json
                messages = await self.aget_messages(conversation_id)
                messages.append(message)
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(messages, indent=2, ensure_ascii=False))
                logger.debug(f"Added message to conversation '{conversation_id}' file (total: {len(messages)}) (async)")
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
    
    async def aclear_conversation(self, conversation_id: str) -> None:
        """Async version of clear_conversation."""
        if not AIOFILES_AVAILABLE:
            # Fallback to sync version in thread pool
            await asyncio.to_thread(self.clear_conversation, conversation_id)
            return
        
        file_path = self._get_file_path(conversation_id)
        
        if await aiofiles.os.path.exists(file_path):
            try:
                await aiofiles.os.remove(file_path)
                logger.info(f"Cleared conversation '{conversation_id}' (deleted file) (async)")
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

        file_extension = f".{self.file_format}"
        conversations.extend(
            filename[:-len(file_extension)]
            for filename in os.listdir(self.storage_dir)
            if filename.endswith(file_extension)
        )
        return conversations
    
    async def alist_conversation_ids(self) -> List[str]:
        """Async version of list_conversation_ids."""
        if not AIOFILES_AVAILABLE:
            # Fallback to sync version in thread pool
            return await asyncio.to_thread(self.list_conversation_ids)
        
        conversations: List[str] = []

        if not await aiofiles.os.path.exists(self.storage_dir):
            return conversations

        try:
            file_extension = f".{self.file_format}"
            # aiofiles doesn't have listdir, so we use asyncio.to_thread for this
            files = await asyncio.to_thread(os.listdir, self.storage_dir)
            conversations.extend(
                filename[:-len(file_extension)]
                for filename in files
                if filename.endswith(file_extension)
            )
        except OSError as e:
            logger.error(f"Error listing conversation files: {e}")
            
        return conversations
    
    def add_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Add multiple messages to a conversation."""
        if not messages:
            return

        processed_messages = []
        for message in messages:
            if "timestamp" not in message:
                m = message.copy()
                m["timestamp"] = datetime.now(timezone.utc).isoformat()
                processed_messages.append(m)
            else:
                processed_messages.append(message)
        
        if self.file_format == "jsonl":
            content_to_write = "\n".join(json.dumps(m, ensure_ascii=False) for m in processed_messages) + "\n"
            self._append_messages_sync(conversation_id, content_to_write)
        else: # json
            existing_messages = self.get_messages(conversation_id)
            all_messages = existing_messages + processed_messages
            self._write_messages_sync(conversation_id, all_messages)

    async def aadd_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Optimized async version of add_messages that appends all new messages at once.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store
        """
        if not messages:
            return
        
        # Add timestamps to messages that don't have them
        processed_messages = []
        for message in messages:
            if "timestamp" not in message:
                m = message.copy()
                m["timestamp"] = datetime.now(timezone.utc).isoformat()
                processed_messages.append(m)
            else:
                processed_messages.append(message)

        if not AIOFILES_AVAILABLE:
            # Fallback to sync version in thread pool
            await asyncio.to_thread(self.add_messages, conversation_id, processed_messages)
            return
        
        file_path = self._get_file_path(conversation_id)
        try:
            if self.file_format == "jsonl":
                content_to_write = "\n".join(json.dumps(m, ensure_ascii=False) for m in processed_messages) + "\n"
                async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                    await f.write(content_to_write)
                logger.debug(f"Appended {len(messages)} messages to conversation '{conversation_id}' file (async batch)")
            else: # json
                existing_messages = await self.aget_messages(conversation_id)
                all_messages = existing_messages + processed_messages
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(all_messages, indent=2, ensure_ascii=False))
                logger.debug(f"Wrote {len(all_messages)} total messages to conversation '{conversation_id}' file (async batch)")

        except IOError as e:
            logger.error(f"Error writing to conversation file '{file_path}': {e}")
            raise

    def _append_messages_sync(self, conversation_id: str, content_to_write: str) -> None:
        """Helper to append a string of messages to a file."""
        file_path = self._get_file_path(conversation_id)
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content_to_write)
        except IOError as e:
            logger.error(f"Error appending messages to file '{file_path}': {e}")
            raise
    
    def _write_messages_sync(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Helper to write a list of messages to a JSON file."""
        file_path = self._get_file_path(conversation_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error writing messages to file '{file_path}': {e}")
            raise

    def list_conversations(self) -> List[ConversationInfo]:
        """List all conversations with metadata by inspecting files."""
        conversation_infos: List[ConversationInfo] = []
        
        for conversation_id in self.list_conversation_ids():
            file_path = self._get_file_path(conversation_id)
            try:
                stat_result = os.stat(file_path)
                
                messages = self.get_messages(conversation_id)
                
                message_count = len(messages)
                last_message = messages[-1] if messages else None
                first_message = messages[0] if messages else None

                info: ConversationInfo = {
                    "conversation_id": conversation_id,
                    "message_count": message_count,
                    "last_message": last_message,
                    "created_at": first_message.get("timestamp") if first_message else datetime.fromtimestamp(stat_result.st_ctime).isoformat(),
                    "updated_at": last_message.get("timestamp") if last_message else datetime.fromtimestamp(stat_result.st_mtime).isoformat(),
                }
                conversation_infos.append(info)
            except (IOError, IndexError, json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not retrieve metadata for conversation '{conversation_id}': {e}")

        conversation_infos.sort(
            key=lambda x: (x["updated_at"] or "", x["conversation_id"]), 
            reverse=True
        )
        
        return conversation_infos

    async def alist_conversations(self) -> List[ConversationInfo]:
        """Async version of list_conversations."""
        if not AIOFILES_AVAILABLE:
            return await asyncio.to_thread(self.list_conversations)

        conversation_infos: List[ConversationInfo] = []
        
        conversation_ids = await self.alist_conversation_ids()
        
        async def get_info(conversation_id: str) -> Optional[ConversationInfo]:
            file_path = self._get_file_path(conversation_id)
            try:
                stat_result = await aiofiles.os.stat(file_path)
                
                messages = await self.aget_messages(conversation_id)
                
                message_count = len(messages)
                last_message = messages[-1] if messages else None
                first_message = messages[0] if messages else None

                info: ConversationInfo = {
                    "conversation_id": conversation_id,
                    "message_count": message_count,
                    "last_message": last_message,
                    "created_at": first_message.get("timestamp") if first_message else datetime.fromtimestamp(stat_result.st_ctime).isoformat(),
                    "updated_at": last_message.get("timestamp") if last_message else datetime.fromtimestamp(stat_result.st_mtime).isoformat(),
                }
                return info
            except (IOError, IndexError, json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not retrieve metadata for conversation '{conversation_id}' (async): {e}")
                return None

        tasks = [get_info(cid) for cid in conversation_ids]
        results = await asyncio.gather(*tasks)
        
        conversation_infos = [info for info in results if info is not None]

        conversation_infos.sort(
            key=lambda x: (x.get("updated_at") or "", x.get("conversation_id")),
            reverse=True
        )
        
        return conversation_infos
    
    def save_as_json(self, conversation_id: str, target_file_path: Optional[str] = None) -> None:
        """
        Reads a conversation and saves it as a standard JSON file.
        
        Args:
            conversation_id: The ID of the conversation to convert.
            target_file_path: The path to save the new JSON file. 
                              If None, it defaults to the same name as the original
                              but with a .json extension.
        """
        messages = self.get_messages(conversation_id)
        if not messages:
            logger.warning(f"No messages found for conversation '{conversation_id}'. Nothing to save.")
            return

        if target_file_path is None:
            safe_id = "".join(c for c in conversation_id if c.isalnum() or c in ('-', '_', '.'))
            target_file_path = os.path.join(self.storage_dir, f"{safe_id}.json")

        try:
            with open(target_file_path, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved conversation '{conversation_id}' to '{target_file_path}'")
        except IOError as e:
            logger.error(f"Error writing to JSON file '{target_file_path}': {e}")
            raise

    async def asave_as_json(self, conversation_id: str, target_file_path: Optional[str] = None) -> None:
        """
        Async version of save_as_json.
        """
        messages = await self.aget_messages(conversation_id)
        if not messages:
            logger.warning(f"No messages found for conversation '{conversation_id}'. Nothing to save. (async)")
            return

        if target_file_path is None:
            safe_id = "".join(c for c in conversation_id if c.isalnum() or c in ('-', '_', '.'))
            target_file_path = os.path.join(self.storage_dir, f"{safe_id}.json")

        if not AIOFILES_AVAILABLE:
            await asyncio.to_thread(self.save_as_json, conversation_id, target_file_path)
            return

        try:
            async with aiofiles.open(target_file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(messages, indent=2, ensure_ascii=False))
            logger.info(f"Successfully saved conversation '{conversation_id}' to '{target_file_path}' (async)")
        except IOError as e:
            logger.error(f"Error writing to JSON file '{target_file_path}': {e}")
            raise 