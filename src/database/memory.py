from typing import ClassVar, Dict, Any, List
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage


class CustomPostgresMemory(BaseChatMemory):
    # Добавляем ClassVar для указания, что это не поле модели, а атрибут класса
    memory_key: ClassVar[str] = "chat_history"

    def __init__(self, storage, user_id: str, chat_id: str):
        super().__init__()
        self.storage = storage
        self.user_id = user_id
        self.chat_id = chat_id

        if not self.storage.get_session(user_id, chat_id):
            self.storage.create_session(user_id, chat_id)

    @property
    def memory_variables(self) -> List[str]:
        """Возвращает список переменных памяти"""
        return [self.memory_key]

    @property
    def chat_memory(self):
        return None

    def load_memory_variables(self, inputs: dict) -> dict:
        messages = self.storage.get_messages(self.user_id, self.chat_id)
        return {
            self.memory_key: [
                HumanMessage(content=m['content']) if m['role'] == "user" else AIMessage(content=m['content'])
                for m in messages
            ]
        }

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self.storage.save_message(self.user_id, self.chat_id, "user", inputs['input'])
        self.storage.save_message(self.user_id, self.chat_id, "ai", outputs['output'])
        self.storage.update_last_interaction(self.user_id, self.chat_id)

    def clear(self) -> None:
        """Очистка памяти"""
        # self.storage.clear_messages(self.user_id, self.chat_id)
        pass