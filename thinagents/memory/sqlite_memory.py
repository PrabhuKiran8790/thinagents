import sqlite3
import json
import logging
from typing import Any, Dict, List, Optional
from thinagents.memory.base_memory import BaseMemory

logger = logging.getLogger(__name__)

class SQLiteMemory(BaseMemory):
    """
    Memory implementation using SQLite to store conversation history.

    This class stores conversations and messages in an SQLite database.
    Messages are stored as JSON strings. It does not provide specific
    support for storing artifacts beyond what is included in the message
    dictionaries themselves.
    """

    def __init__(self, db_path: str, table_name: str = "main"):
        """
        Initialize SQLiteMemory.

        Args:
            db_path: Path to the SQLite database file. 
                     If ":memory:", an in-memory database will be used.
            table_name: Prefix for table names. Tables will be named 
                       '{table_name}_conversations' and '{table_name}_messages'.
                       Defaults to "main".
        """
        self.db_path = db_path
        self.table_name = table_name
        self.conversations_table = f"{table_name}_conversations"
        self.messages_table = f"{table_name}_messages"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establish and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize the database and create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.conversations_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL
            )
            """)
            
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_ref_id INTEGER NOT NULL,
                message_json TEXT NOT NULL,
                timestamp TEXT,  -- Store as ISO format string, from message content for ordering
                FOREIGN KEY (conversation_ref_id) REFERENCES {self.conversations_table} (id) ON DELETE CASCADE
            )
            """)
            
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.conversations_table}_conversation_id ON {self.conversations_table} (conversation_id);
            """)
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.messages_table}_conversation_ref_id ON {self.messages_table} (conversation_ref_id);
            """)
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.messages_table}_timestamp ON {self.messages_table} (timestamp);
            """)
            conn.commit()

    def _get_conversation_db_id(self, conversation_id: str, create_if_not_exists: bool = True) -> Optional[int]:
        """Get the internal DB ID for a conversation_id. Optionally create if not found."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id FROM {self.conversations_table} WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
            elif create_if_not_exists:
                cursor.execute(f"INSERT INTO {self.conversations_table} (conversation_id) VALUES (?)", (conversation_id,))
                conn.commit()
                return cursor.lastrowid
            return None

    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Store a new message in the conversation history.
        """
        conv_db_id = self._get_conversation_db_id(conversation_id, create_if_not_exists=True)
        if conv_db_id is None:
            # This case should ideally not be reached if create_if_not_exists is True
            # and insert is successful.
            logger.error(f"Failed to get or create conversation DB ID for {conversation_id}. Message not added.")
            return

        message_json = json.dumps(message)
        
        msg_timestamp_str: Optional[str] = None
        if "timestamp" in message and isinstance(message["timestamp"], str):
            msg_timestamp_str = message["timestamp"]
        # Add more sophisticated timestamp handling if needed (e.g., datetime objects to ISO string)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO {self.messages_table} (conversation_ref_id, message_json, timestamp) VALUES (?, ?, ?)",
                (conv_db_id, message_json, msg_timestamp_str)
            )
            conn.commit()

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a given conversation ID.
        Messages are returned in chronological order based on their 'timestamp'
        field (if present and sortable), otherwise by insertion order (message.id).
        """
        conv_db_id = self._get_conversation_db_id(conversation_id, create_if_not_exists=False)
        if conv_db_id is None:
            return [] # No such conversation exists

        messages_list: List[Dict[str, Any]] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Order by our stored timestamp (extracted from message), then by message ID for stable sort.
            # NULL timestamps will typically sort first in ASC order with SQLite.
            cursor.execute(
                f"SELECT message_json FROM {self.messages_table} WHERE conversation_ref_id = ? ORDER BY timestamp ASC, id ASC",
                (conv_db_id,)
            )
            for row in cursor.fetchall():
                try:
                    messages_list.append(json.loads(row[0]))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message JSON for conversation {conversation_id}: {e} - Data: {row[0][:100]}...")
        return messages_list

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear all messages for a specific conversation.
        The conversation entry itself in the conversations table is not removed,
        allowing the conversation_id to remain listed.
        """
        conv_db_id = self._get_conversation_db_id(conversation_id, create_if_not_exists=False)
        if conv_db_id is None:
            logger.debug(f"Conversation ID '{conversation_id}' not found, nothing to clear.")
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.messages_table} WHERE conversation_ref_id = ?", (conv_db_id,))
            conn.commit()
            logger.info(f"Cleared all messages for conversation ID '{conversation_id}' (DB ID: {conv_db_id}).")

    def list_conversation_ids(self) -> List[str]:
        """
        List all conversation IDs in the memory store.
        """
        ids_list: List[str] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT conversation_id FROM {self.conversations_table} ORDER BY conversation_id ASC")
            ids_list.extend(row[0] for row in cursor.fetchall())
        return ids_list