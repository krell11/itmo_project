import os
import json
import tempfile
import re
from typing import Dict, List, Optional, Any, Union
import logging

import glob

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema import AgentAction, AgentFinish

# from ..json_parsing.json_creator import ConfigFactory
from ..json_parsing.new_json_creator import StableConfigFactory as ConfigFactory
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class ROMAssistantAgent:
    def __init__(
            self,
            llm_api_base: str,
            llm_model_name: str,
            chroma_db_path: str,
            postgres_connection_string: str
    ):

        self.llm = ChatOpenAI(
            base_url=llm_api_base,
            model_name=llm_model_name,
            openai_api_key="dummy_key",
            temperature=0,
            max_tokens=1000,
            stop=["\nObservation:", "Observation:"]
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True})

        if os.path.exists(chroma_db_path):
            self.vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=self.embeddings
            )
        else:
            os.makedirs(os.path.dirname(chroma_db_path), exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=self.embeddings
            )

        self.postgres_connection_string = postgres_connection_string
        self.tools = self._init_tools()
        self.agent = self._init_agent()

    def _init_tools(self) -> List[Tool]:
        """Инициализация инструментов для агента"""
        return [
            Tool(
                name="create_config",
                func=self._create_config,
                description="Создание конфигурации ROM-модуля на основе описания задачи"
            ),
            Tool(
                name="run_rom_module",
                func=self._run_rom_module,
                description="Запуск ROM-модуля с файлом конфигурации"
            ),
            # Tool(
            #     name="read_results_file",
            #     func=self._read_results_file,
            #     description="Чтение файла с результатами расчета ROM-модуля"
            # )
        ]

    def _init_agent(self) -> AgentExecutor:
        """Инициализация агента с create_react_agent"""
        from langchain.agents import create_react_agent

        template = """Отвечай на вопросы о ROM-модулях на русском языке.

        Инструменты:
        {tools}
        
        Формат ответа:
        Question: {input}
        Thought: мои размышления на русском
        Action: [выбери один из: {tool_names}]
        Action Input: входной_параметр_без_кавычек_и_равно
        Observation: результат инструмента
        Thought: анализ результата на русском
        Final Answer: итоговый ответ на русском
        
        ВАЖНЫЕ ПРАВИЛА для Action Input:
        - Пиши ТОЛЬКО значение параметра
        - НЕ используй кавычки или знак равно
        - Для run_rom_module пиши: E:\\path\\to\\file.json
        - Для create_config пиши: описание параметров
        
        Question: {input}
        {agent_scratchpad}"""
        old = """        
        - Для read_results_file пиши: E:\\path\\to\\directory"""
        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            max_execution_time=360,
            handle_parsing_errors="Ошибка формата. Action Input должен содержать только значение без кавычек и знака равно.",
            return_intermediate_steps=True
        )

    def _create_config(self, query: str) -> str:
        """Создание конфигурации с помощью ConfigFactory"""
        try:
            config = ConfigFactory.create_config(
                user_input=query,
                script_path="/path/to/default/script.txt",
                mode=None,
                bat_path=None,
                results_file="result.txt",
                project_path="/path/to/default/project",
                model_path=None
            )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
                json.dump(config, temp_file, indent=2, ensure_ascii=False)
                config_path = temp_file.name

            logger.info(f"Created configuration: {config_path}")
            return f"Конфигурация создана: {config_path}"
        except Exception as e:
            logger.error(f"Error creating config: {str(e)}")
            return f"Ошибка создания конфигурации: {str(e)}"

    def _run_rom_module(self, config_path: str) -> str:
        """Запуск ROM-модуля с указанной конфигурацией"""
        try:
            config_path = config_path.strip().strip('"\'')

            if not os.path.exists(config_path):
                return f"ОШИБКА: файл {config_path} не найден"

            logger.info(f"Running ROM module with: {config_path}")

            from ..example import main as rom_main
            rom_main(config_path)

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            project_path = config.get("project", {}).get("project_path", "")

            directory_path = project_path.strip().strip('"\'')

            if not os.path.exists(directory_path):
                return f"ОШИБКА: папка {directory_path} не найдена"

            result_patterns = ['results_*', 'result.txt', 'output.txt']
            result_files = []

            for pattern in result_patterns:
                files = glob.glob(os.path.join(directory_path, pattern))
                result_files.extend(files)

            if not result_files:
                return f"Файлы результатов не найдены в {directory_path}"

            result_file = result_files[0]

            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            return f"результаты выполнения ROM модуля : {content}"
        except Exception as e:
            logger.error(f"Error running ROM: {str(e)}")
            return f"ОШИБКА запуска: {str(e)}"

    def _read_results_file(self, directory_path: str) -> str:
        """Чтение файла с результатами"""
        try:

            directory_path = directory_path.strip().strip('"\'')

            if not os.path.exists(directory_path):
                return f"ОШИБКА: папка {directory_path} не найдена"

            result_patterns = ['results_*', 'result.txt', 'output.txt']
            result_files = []

            for pattern in result_patterns:
                files = glob.glob(os.path.join(directory_path, pattern))
                result_files.extend(files)

            if not result_files:
                return f"Файлы результатов не найдены в {directory_path}"

            result_file = result_files[0]

            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            return f"Результаты из {os.path.basename(result_file)}:\n{content}"

        except Exception as e:
            logger.error(f"Error reading results: {str(e)}")
            return f"ОШИБКА чтения: {str(e)}"

    def _get_chat_memory(self, session_id: str) -> PostgresChatMessageHistory:
        """Получение истории чата из PostgreSQL"""
        try:
            return PostgresChatMessageHistory(
                connection_string=self.postgres_connection_string,
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error creating chat history: {str(e)}")
            return PostgresChatMessageHistory(
                connection_string=self.postgres_connection_string,
                session_id=session_id
            )

    def process_query(self, user_id: str, chat_id: str, query: str,
                      config: Optional[Dict] = None, config_path: Optional[str] = None) -> str:
        """
        Обработка запроса пользователя
        """
        try:
            session_id = f"{user_id}_{chat_id}"

            enhanced_query = query

            if config_path and os.path.exists(config_path):
                enhanced_query = f"{query}\n\nГотовый конфиг: {config_path}"
                logger.info(f"Using config file: {config_path}")
            elif config:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
                    json.dump(config, temp_file, indent=2, ensure_ascii=False)
                    temp_config_path = temp_file.name
                enhanced_query = f"{query}\n\nГотовый конфиг: {temp_config_path}"
                logger.info(f"Saved config to: {temp_config_path}")

            result = self.agent.invoke({"input": enhanced_query})

            chat_history = self._get_chat_memory(session_id)
            chat_history.add_user_message(query)
            chat_history.add_ai_message(result["output"])

            return result["output"]

        except Exception as e:
            error_message = f"Ошибка: {str(e)}"
            logger.error(f"Error in process_query: {str(e)}")

            try:
                session_id = f"{user_id}_{chat_id}"
                chat_history = self._get_chat_memory(session_id)
                chat_history.add_user_message(query)
                chat_history.add_ai_message(error_message)
            except Exception as save_error:
                logger.error(f"Error saving to history: {str(save_error)}")

            return error_message
