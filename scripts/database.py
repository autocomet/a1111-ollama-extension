#!/usr/bin/env python3
"""
A1111 Ollama Extension - Database Module

Handles local database operations for storing conversations, prompts, and settings.
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class OllamaDatabase:
    """Database handler for the Ollama extension."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database connection."""
        if db_path is None:
            # Default to a database in the extension directory
            extension_dir = Path(__file__).parent.parent
            db_path = extension_dir / "ollama_extension.db"
        
        self.db_path = str(db_path)
        self.connection = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Create conversations table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ollama_response TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            
            # Create prompts table for prompt suggestions
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_text TEXT NOT NULL UNIQUE,
                    category TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create settings table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create models table for tracking available models
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    size TEXT,
                    last_used TEXT,
                    is_available BOOLEAN DEFAULT 1
                )
            """)
            
            self.connection.commit()
            print("Database initialized successfully")
            
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise
    
    def save_conversation(self, user_message: str, ollama_response: str, 
                         model_name: str, session_id: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> int:
        """Save a conversation to the database."""
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (timestamp, user_message, ollama_response, model_name, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, user_message, ollama_response, model_name, session_id, metadata_json))
            
            conversation_id = cursor.lastrowid
            self.connection.commit()
            return conversation_id
            
        except sqlite3.Error as e:
            print(f"Error saving conversation: {e}")
            return -1
    
    def get_conversation_history(self, limit: int = 50, 
                               model_name: Optional[str] = None) -> List[Dict]:
        """Retrieve conversation history."""
        try:
            cursor = self.connection.cursor()
            
            if model_name:
                cursor.execute("""
                    SELECT id, timestamp, user_message, ollama_response, model_name, session_id
                    FROM conversations 
                    WHERE model_name = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (model_name, limit))
            else:
                cursor.execute("""
                    SELECT id, timestamp, user_message, ollama_response, model_name, session_id
                    FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            conversations = []
            
            for row in rows:
                conversations.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'user_message': row[2],
                    'ollama_response': row[3],
                    'model_name': row[4],
                    'session_id': row[5]
                })
            
            return conversations
            
        except sqlite3.Error as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    def save_prompt(self, prompt_text: str, category: Optional[str] = None) -> bool:
        """Save a prompt for future suggestions."""
        timestamp = datetime.now().isoformat()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO prompts (prompt_text, category, created_at)
                VALUES (?, ?, ?)
            """, (prompt_text, category, timestamp))
            
            # Update usage count if prompt already exists
            cursor.execute("""
                UPDATE prompts 
                SET usage_count = usage_count + 1, last_used = ?
                WHERE prompt_text = ?
            """, (timestamp, prompt_text))
            
            self.connection.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error saving prompt: {e}")
            return False
    
    def get_prompt_suggestions(self, partial_prompt: str, limit: int = 10) -> List[str]:
        """Get prompt suggestions based on partial input."""
        try:
            cursor = self.connection.cursor()
            search_term = f"%{partial_prompt}%"
            
            cursor.execute("""
                SELECT prompt_text 
                FROM prompts 
                WHERE prompt_text LIKE ?
                ORDER BY usage_count DESC, last_used DESC
                LIMIT ?
            """, (search_term, limit))
            
            rows = cursor.fetchall()
            return [row[0] for row in rows]
            
        except sqlite3.Error as e:
            print(f"Error getting prompt suggestions: {e}")
            return []
    
    def get_setting(self, key: str, default_value: Optional[str] = None) -> Optional[str]:
        """Get a setting value from the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else default_value
            
        except sqlite3.Error as e:
            print(f"Error getting setting {key}: {e}")
            return default_value
    
    def set_setting(self, key: str, value: str) -> bool:
        """Set a setting value in the database."""
        timestamp = datetime.now().isoformat()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, timestamp))
            
            self.connection.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error setting {key}: {e}")
            return False
    
    def update_model_info(self, name: str, description: str = None, 
                         size: str = None, is_available: bool = True) -> bool:
        """Update model information in the database."""
        timestamp = datetime.now().isoformat()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO models (name, description, size, last_used, is_available)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, size, timestamp, is_available))
            
            self.connection.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error updating model info: {e}")
            return False
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models from the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT name, description, size, last_used 
                FROM models 
                WHERE is_available = 1
                ORDER BY last_used DESC
            """)
            
            rows = cursor.fetchall()
            models = []
            
            for row in rows:
                models.append({
                    'name': row[0],
                    'description': row[1],
                    'size': row[2],
                    'last_used': row[3]
                })
            
            return models
            
        except sqlite3.Error as e:
            print(f"Error getting available models: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """Clean up old conversation data."""
        try:
            cutoff_date = datetime.now().replace(day=datetime.now().day - days_to_keep).isoformat()
            
            cursor = self.connection.cursor()
            cursor.execute("""
                DELETE FROM conversations 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            print(f"Cleaned up {deleted_count} old conversation records")
            return True
            
        except sqlite3.Error as e:
            print(f"Error cleaning up old data: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __del__(self):
        """Ensure database connection is closed on object destruction."""
        self.close()
