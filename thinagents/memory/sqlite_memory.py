import asyncio
import sqlite3
import json
import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager, asynccontextmanager
from threading import Lock, local
from thinagents.memory.base_memory import BaseMemory, ConversationInfo

logger = logging.getLogger(__name__)

# Check if aiosqlite is available
try:
    import aiosqlite  # type: ignore
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

_thread_local = local()

def get_sync_db_connection(db_path: str) -> sqlite3.Connection:
    """Get a thread-local synchronous database connection."""
    if not hasattr(_thread_local, "sqlite_connection"):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        _thread_local.sqlite_connection = conn
    return _thread_local.sqlite_connection

@contextmanager
def managed_sync_connection(db_path: str):
    """Context manager for synchronous database connections."""
    conn = get_sync_db_connection(db_path)
    try:
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        # In a real app, you might want more sophisticated error handling,
        # like rolling back a transaction.
        raise

class SQLiteMemory(BaseMemory):
    """
    Memory implementation using SQLite to store conversation history.

    This class stores conversations and messages in an SQLite database.
    It uses thread-local connections for synchronous operations and
    on-demand connections for asynchronous operations.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLiteMemory.

        Args:
            db_path: Path to the SQLite database file. 
                     If ":memory:", an in-memory database will be used.
        """
        self.db_path = db_path
        
        # Initialize database
        self._init_db()
        
        # Warn if aiosqlite is not available
        if not AIOSQLITE_AVAILABLE:
            logger.warning("aiosqlite not available. Async operations will use sync methods in thread pool.")
    
    def _init_db(self) -> None:
        """Initialize the database and create tables if they don't exist."""
        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_ref_id INTEGER NOT NULL,
                message_json TEXT NOT NULL,
                timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_ref_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations (conversation_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_ref_id ON messages (conversation_ref_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages (created_at);
            """)
            
            # Create trigger to update updated_at
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_conversation_timestamp
            AFTER INSERT ON messages
            BEGIN
                UPDATE conversations 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = NEW.conversation_ref_id;
            END;
            """)
            
            conn.commit()

    def _get_conversation_db_id(self, conversation_id: str, create_if_not_exists: bool = True) -> Optional[int]:
        """Get the internal DB ID for a conversation_id. Optionally create if not found."""
        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM conversations WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
            elif create_if_not_exists:
                cursor.execute("INSERT INTO conversations (conversation_id) VALUES (?)", (conversation_id,))
                conn.commit()
                return cursor.lastrowid
            return None

    async def _aget_conversation_db_id(self, conversation_id: str, create_if_not_exists: bool = True) -> Optional[int]:
        """Async version of _get_conversation_db_id."""
        if not AIOSQLITE_AVAILABLE:
            return await asyncio.to_thread(self._get_conversation_db_id, conversation_id, create_if_not_exists)
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            cursor = await conn.cursor()
            await cursor.execute("SELECT id FROM conversations WHERE conversation_id = ?", (conversation_id,))
            row = await cursor.fetchone()
            if row:
                return row[0]
            elif create_if_not_exists:
                await cursor.execute("INSERT INTO conversations (conversation_id) VALUES (?)", (conversation_id,))
                await conn.commit()
                return cursor.lastrowid
            return None

    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Store a new message in the conversation history.
        """
        conv_db_id = self._get_conversation_db_id(conversation_id, create_if_not_exists=True)
        if conv_db_id is None:
            logger.error(f"Failed to get or create conversation DB ID for {conversation_id}. Message not added.")
            return

        message_json = json.dumps(message)
        
        msg_timestamp_str: Optional[str] = None
        if "timestamp" in message and isinstance(message["timestamp"], str):
            msg_timestamp_str = message["timestamp"]

        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (conversation_ref_id, message_json, timestamp) VALUES (?, ?, ?)",
                (conv_db_id, message_json, msg_timestamp_str)
            )
            conn.commit()

    async def aadd_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Async version of add_message.
        """
        if not AIOSQLITE_AVAILABLE:
            await asyncio.to_thread(self.add_message, conversation_id, message)
            return
        
        conv_db_id = await self._aget_conversation_db_id(conversation_id, create_if_not_exists=True)
        if conv_db_id is None:
            logger.error(f"Failed to get or create conversation DB ID for {conversation_id}. Message not added.")
            return

        message_json = json.dumps(message)
        
        msg_timestamp_str: Optional[str] = None
        if "timestamp" in message and isinstance(message["timestamp"], str):
            msg_timestamp_str = message["timestamp"]

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO messages (conversation_ref_id, message_json, timestamp) VALUES (?, ?, ?)",
                (conv_db_id, message_json, msg_timestamp_str)
            )
            await conn.commit()

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a given conversation ID.
        Messages are returned in chronological order based on their 'timestamp'
        field (if present and sortable), otherwise by insertion order (message.id).
        """
        conv_db_id = self._get_conversation_db_id(conversation_id, create_if_not_exists=False)
        if conv_db_id is None:
            return []

        messages_list: List[Dict[str, Any]] = []
        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_json FROM messages WHERE conversation_ref_id = ? ORDER BY timestamp ASC, id ASC",
                (conv_db_id,)
            )
            for row in cursor.fetchall():
                try:
                    messages_list.append(json.loads(row[0]))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message JSON for conversation {conversation_id}: {e} - Data: {row[0][:100]}...")
        return messages_list

    async def aget_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Async version of get_messages.
        """
        if not AIOSQLITE_AVAILABLE:
            return await asyncio.to_thread(self.get_messages, conversation_id)
        
        conv_db_id = await self._aget_conversation_db_id(conversation_id, create_if_not_exists=False)
        if conv_db_id is None:
            return []

        messages_list: List[Dict[str, Any]] = []
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT message_json FROM messages WHERE conversation_ref_id = ? ORDER BY timestamp ASC, id ASC",
                (conv_db_id,)
            )
            async for row in cursor:
                try:
                    messages_list.append(json.loads(row[0]))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message JSON for conversation {conversation_id}: {e} - Data: {row[0][:100]}...")
        return messages_list

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear all messages for a specific conversation.
        The conversation entry itself in the 'conversations' table is not removed,
        allowing the conversation_id to remain listed.
        """
        conv_db_id = self._get_conversation_db_id(conversation_id, create_if_not_exists=False)
        if conv_db_id is None:
            logger.debug(f"Conversation ID '{conversation_id}' not found, nothing to clear.")
            return

        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_ref_id = ?", (conv_db_id,))
            conn.commit()
            logger.info(f"Cleared all messages for conversation ID '{conversation_id}' (DB ID: {conv_db_id}).")

    async def aclear_conversation(self, conversation_id: str) -> None:
        """
        Async version of clear_conversation.
        """
        if not AIOSQLITE_AVAILABLE:
            await asyncio.to_thread(self.clear_conversation, conversation_id)
            return
        
        conv_db_id = await self._aget_conversation_db_id(conversation_id, create_if_not_exists=False)
        if conv_db_id is None:
            logger.debug(f"Conversation ID '{conversation_id}' not found, nothing to clear.")
            return

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            cursor = await conn.cursor()
            await cursor.execute("DELETE FROM messages WHERE conversation_ref_id = ?", (conv_db_id,))
            await conn.commit()
            logger.info(f"Cleared all messages for conversation ID '{conversation_id}' (DB ID: {conv_db_id}) (async).")

    def list_conversation_ids(self) -> List[str]:
        """
        List all conversation IDs in the memory store.
        """
        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT conversation_id FROM conversations ORDER BY created_at DESC")
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    async def alist_conversation_ids(self) -> List[str]:
        """
        Async version of list_conversation_ids.
        """
        if not AIOSQLITE_AVAILABLE:
            return await asyncio.to_thread(self.list_conversation_ids)
        
        ids_list: List[str] = []
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            cursor = await conn.cursor()
            await cursor.execute("SELECT conversation_id FROM conversations ORDER BY updated_at DESC, conversation_id ASC")
            async for row in cursor:
                ids_list.append(row[0])
        return ids_list

    def list_conversations(self) -> List[ConversationInfo]:
        """List all conversations with metadata using an efficient query."""
        query = """
        SELECT
            c.conversation_id,
            COUNT(m.id) as message_count,
            (SELECT message_json FROM messages WHERE conversation_ref_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message,
            MIN(c.created_at) as created_at,
            MAX(c.updated_at) as updated_at
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_ref_id
        GROUP BY c.id
        ORDER BY updated_at DESC;
        """
        with managed_sync_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

        infos = []
        for row in rows:
            last_message_json = row[2]
            info: ConversationInfo = {
                "conversation_id": row[0],
                "message_count": row[1],
                "last_message": json.loads(last_message_json) if last_message_json else None,
                "created_at": row[3],
                "updated_at": row[4],
            }
            infos.append(info)
        
        return infos

    async def alist_conversations(self) -> List[ConversationInfo]:
        """Async version of list_conversations."""
        if not AIOSQLITE_AVAILABLE:
            return await asyncio.to_thread(self.list_conversations)

        query = """
        SELECT
            c.conversation_id,
            COUNT(m.id) as message_count,
            (SELECT message_json FROM messages WHERE conversation_ref_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message,
            MIN(c.created_at) as created_at,
            MAX(c.updated_at) as updated_at
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_ref_id
        GROUP BY c.id
        ORDER BY updated_at DESC;
        """
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(query)
            rows = await cursor.fetchall()

        infos = []
        for row in rows:
            last_message_json = row[2]
            info: ConversationInfo = {
                "conversation_id": row[0],
                "message_count": row[1],
                "last_message": json.loads(last_message_json) if last_message_json else None,
                "created_at": row[3],
                "updated_at": row[4],
            }
            infos.append(info)
        
        return infos

    # Optimized batch operations
    async def aadd_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Optimized async version of add_messages that uses batch inserts.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store
        """
        if not messages:
            return
        
        if not AIOSQLITE_AVAILABLE:
            await asyncio.to_thread(self.add_messages, conversation_id, messages)
            return
        
        conv_db_id = await self._aget_conversation_db_id(conversation_id, create_if_not_exists=True)
        if conv_db_id is None:
            logger.error(f"Failed to get or create conversation DB ID for {conversation_id}. Messages not added.")
            return

        # Prepare batch data
        batch_data = []
        for message in messages:
            message_json = json.dumps(message)
            msg_timestamp_str = message.get("timestamp") if isinstance(message.get("timestamp"), str) else None
            batch_data.append((conv_db_id, message_json, msg_timestamp_str))

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            cursor = await conn.cursor()
            await cursor.executemany(
                "INSERT INTO messages (conversation_ref_id, message_json, timestamp) VALUES (?, ?, ?)",
                batch_data
            )
            await conn.commit()
            logger.debug(f"Added {len(messages)} messages to conversation '{conversation_id}' (batch async)")

    def close(self) -> None:
        """Close the thread-local database connection."""
        if hasattr(_thread_local, "sqlite_connection"):
            _thread_local.sqlite_connection.close()
            del _thread_local.sqlite_connection

    def __del__(self):
        """Ensure the connection is closed when the object is destroyed."""
        self.close()