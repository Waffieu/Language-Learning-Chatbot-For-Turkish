import os
import json
import logging
import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Set
from functools import lru_cache
import config

# Configure logging
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self):
        # Dictionary to store conversations by chat_id
        self.conversations: Dict[int, List[Dict[str, str]]] = {}

        # Memory cache to reduce disk I/O
        self.memory_cache: Dict[int, Dict[str, Any]] = {}

        # Set to track modified conversations that need saving
        self.modified_chats: Set[int] = set()

        # Thread lock for thread safety
        self.lock = threading.RLock()

        # Auto-save timer
        self.last_save_time = time.time()
        self.save_interval = 60  # seconds

        # Create memory directory if it doesn't exist
        os.makedirs(config.MEMORY_DIR, exist_ok=True)

        # Load existing memories
        self._load_all_memories()

        # Start background auto-save thread
        self.running = True
        self.save_thread = threading.Thread(target=self._auto_save_thread, daemon=True)
        self.save_thread.start()

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        """
        Add a message to the conversation history for a specific chat

        Args:
            chat_id: The Telegram chat ID
            role: Either 'user' or 'model'
            content: The message content
        """
        with self.lock:
            if chat_id not in self.conversations:
                self.conversations[chat_id] = []

            # Create message object
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time()
            }

            # Add message to conversation
            self.conversations[chat_id].append(message)

            # Trim the conversation if it exceeds the long memory size
            if len(self.conversations[chat_id]) > config.LONG_MEMORY_SIZE:
                self.conversations[chat_id] = self.conversations[chat_id][-config.LONG_MEMORY_SIZE:]

            # Mark this chat as modified
            self.modified_chats.add(chat_id)

            # Clear any cached results for this chat
            self._clear_cache_for_chat(chat_id)

            # Save immediately if this is the first message
            if len(self.conversations[chat_id]) == 1:
                self._save_memory(chat_id)
            # Otherwise, let the auto-save handle it

    @lru_cache(maxsize=32)
    def get_short_memory(self, chat_id: int) -> List[Dict[str, str]]:
        """
        Get the short-term memory (most recent messages) for a specific chat

        Args:
            chat_id: The Telegram chat ID

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        with self.lock:
            if chat_id not in self.conversations:
                return []

            # Get the most recent messages up to SHORT_MEMORY_SIZE
            messages = self.conversations[chat_id][-config.SHORT_MEMORY_SIZE:]

            # For compatibility with the rest of the code, return only role and content
            return [{'role': msg['role'], 'content': msg['content']} for msg in messages]

    @lru_cache(maxsize=32)
    def get_long_memory(self, chat_id: int) -> List[Dict[str, str]]:
        """
        Get the long-term memory (all stored messages) for a specific chat

        Args:
            chat_id: The Telegram chat ID

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        with self.lock:
            if chat_id not in self.conversations:
                return []

            # For compatibility with the rest of the code, return only role and content
            return [{'role': msg['role'], 'content': msg['content']} for msg in self.conversations[chat_id]]

    def _get_memory_file_path(self, chat_id: int) -> str:
        """
        Get the file path for a specific chat's memory file

        Args:
            chat_id: The Telegram chat ID

        Returns:
            Path to the memory file
        """
        return os.path.join(config.MEMORY_DIR, f"memory_{chat_id}.json")

    def _save_memory(self, chat_id: int) -> None:
        """
        Save a specific chat's memory to disk

        Args:
            chat_id: The Telegram chat ID
        """
        with self.lock:
            if chat_id not in self.conversations:
                return

            try:
                memory_file = self._get_memory_file_path(chat_id)
                temp_file = f"{memory_file}.tmp"

                # First write to a temporary file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.conversations[chat_id], f, ensure_ascii=False, indent=2)

                # Then rename it to the actual file (atomic operation)
                os.replace(temp_file, memory_file)

                # Remove from modified set
                if chat_id in self.modified_chats:
                    self.modified_chats.remove(chat_id)

                logger.debug(f"Saved memory for chat {chat_id} to {memory_file}")
            except Exception as e:
                logger.error(f"Error saving memory for chat {chat_id}: {e}")

    def _load_memory(self, chat_id: int) -> None:
        """
        Load a specific chat's memory from disk

        Args:
            chat_id: The Telegram chat ID
        """
        memory_file = self._get_memory_file_path(chat_id)
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                # Ensure all messages have a timestamp
                for msg in loaded_data:
                    if 'timestamp' not in msg:
                        msg['timestamp'] = time.time()

                with self.lock:
                    self.conversations[chat_id] = loaded_data

                logger.info(f"Loaded memory for chat {chat_id} with {len(self.conversations[chat_id])} messages")
            except Exception as e:
                logger.error(f"Error loading memory for chat {chat_id}: {e}")
                # Initialize empty conversation if loading fails
                with self.lock:
                    self.conversations[chat_id] = []

    def _load_all_memories(self) -> None:
        """
        Load all memories from disk
        """
        try:
            # Get all memory files
            memory_files = [f for f in os.listdir(config.MEMORY_DIR) if f.startswith("memory_") and f.endswith(".json")]
            logger.info(f"Found {len(memory_files)} memory files to load")

            # Load each memory file
            for memory_file in memory_files:
                try:
                    # Extract chat_id from filename (memory_CHATID.json)
                    chat_id = int(memory_file.split('_')[1].split('.')[0])
                    self._load_memory(chat_id)
                except Exception as e:
                    logger.error(f"Error processing memory file {memory_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")

    def _auto_save_thread(self) -> None:
        """
        Background thread that periodically saves modified conversations
        """
        while self.running:
            try:
                # Sleep for a short interval
                time.sleep(5)

                # Check if it's time to save
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self._save_all_modified()
                    self.last_save_time = current_time
            except Exception as e:
                logger.error(f"Error in auto-save thread: {e}")

    def _save_all_modified(self) -> None:
        """
        Save all modified conversations to disk
        """
        with self.lock:
            modified = list(self.modified_chats)

        if modified:
            logger.debug(f"Auto-saving {len(modified)} modified conversations")
            for chat_id in modified:
                self._save_memory(chat_id)

    def _clear_cache_for_chat(self, chat_id: int) -> None:
        """
        Clear cached memory results for a specific chat
        """
        # Clear the LRU cache for this chat
        self.get_short_memory.cache_clear()
        self.get_long_memory.cache_clear()

    def shutdown(self) -> None:
        """
        Properly shut down the memory system, saving any pending changes
        """
        logger.info("Shutting down memory system")
        self.running = False
        if hasattr(self, 'save_thread') and self.save_thread.is_alive():
            self.save_thread.join(timeout=10)
        self._save_all_modified()
        logger.info("Memory system shutdown complete")
