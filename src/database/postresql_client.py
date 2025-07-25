import json
import logging
from typing import Dict, Optional
import psycopg2
from psycopg2.extras import Json, DictCursor


class PostgreSQLStorage:
    def __init__(self, db_config: Dict, logger: Optional[logging.Logger] = None):
        self.db_config = db_config
        self.logger = logger or logging.getLogger("PostgreSQLStorage")
        self._init_database()

    def _get_connection(self):
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise

    def _init_database(self):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            user_id VARCHAR(50),
                            chat_id VARCHAR(50),
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            last_interaction TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            config JSONB,
                            config_path TEXT,
                            model_description TEXT,
                            PRIMARY KEY (user_id, chat_id)
                        )
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS messages (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(50),
                            chat_id VARCHAR(50),
                            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            role VARCHAR(20),
                            content TEXT,
                            FOREIGN KEY (user_id, chat_id) REFERENCES sessions (user_id, chat_id) ON DELETE CASCADE
                        )
                    """)
                conn.commit()
                self.logger.info("PostgreSQL tables ensured")
        except Exception as e:
            self.logger.error(f"DB init failed: {e}")
            raise

    def get_session(self, user_id: str, chat_id: str):
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT * FROM sessions WHERE user_id=%s AND chat_id=%s", (user_id, chat_id))
                return cur.fetchone()

    def create_session(self, user_id: str, chat_id: str):
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("INSERT INTO sessions (user_id, chat_id) VALUES (%s, %s) RETURNING *", (user_id, chat_id))
                conn.commit()
                return cur.fetchone()

    def update_last_interaction(self, user_id: str, chat_id: str):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sessions SET last_interaction = CURRENT_TIMESTAMP
                    WHERE user_id=%s AND chat_id=%s
                """, (user_id, chat_id))
                conn.commit()

    def save_message(self, user_id: str, chat_id: str, role: str, content: str):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (user_id, chat_id, role, content)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, chat_id, role, content))
                conn.commit()

    def get_messages(self, user_id: str, chat_id: str):
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT role, content FROM messages
                    WHERE user_id = %s AND chat_id = %s
                    ORDER BY timestamp ASC
                """, (user_id, chat_id))
                return cur.fetchall()

    def update_session_config(self, user_id: str, chat_id: str,
                              config: Dict,
                              config_path: Optional[str] = None,
                              model_description: Optional[str] = None):
        update_fields = ["config = %s"]
        params = [Json(config)]

        if config_path:
            update_fields.append("config_path = %s")
            params.append(config_path)
        if model_description:
            update_fields.append("model_description = %s")
            params.append(model_description)

        update_fields.append("last_interaction = CURRENT_TIMESTAMP")
        params.extend([user_id, chat_id])

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE sessions
                    SET {', '.join(update_fields)}
                    WHERE user_id = %s AND chat_id = %s
                """, params)
                conn.commit()
