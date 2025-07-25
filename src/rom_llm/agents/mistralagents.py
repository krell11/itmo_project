import json
import os
import glob
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import tempfile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from json import load


def extract_json_from_response(response: str) -> Dict[str, Any]:
    try:
        start = response.find('{')
        end = response.rfind('}') + 1

        if start != -1 and end != 0:
            json_str = response[start:end]
            return json.loads(json_str)

        return {}
    except:
        return {}


class ROMConfigChatAgent:
    def __init__(self, api_key: str):
        self.system_prompt = """
Ты - эксперт по настройке ROM-модуля для создания моделей пониженного порядка и суррогатных моделей. Твоя задача - помочь пользователю создать корректный JSON-файл конфигурации на основе его требований.

## Доступные методы модуля:
1. **SVD** - сингулярное разложение для моделей пониженного порядка (подходит для предсказания полей значений)
2. **NN** - нейронные сети для суррогатных моделей (подходит для предсказания отдельных значений)
3. **GPR** - регрессия гауссовских процессов для суррогатных моделей (подходит для предсказания отдельных значений)

## Режимы работы:
- **TRAIN** - только обучение модели
- **TP** - обучение и последующее предсказание
- **PREDICT** - только предсказание с использованием готовой модели

## Структура JSON конфигурации:

### Обязательные поля для всех методов:

```json
{
  "module": "ROM",
  "project": {
    "project_path": "абсолютный_путь_к_рабочей_директории",
    "script_paths": ["массив_путей_к_макросам"],
    "bat_path": "путь_к_bat_файлу",
    "results_file": "имя_файла_с_результатами пиши result.txt",
    "number_of_cores": число_ядер,
    "del_data": true/false,
    "date": true/false
  },
  "experiment": {
    "mode": "TRAIN/TP/PREDICT",
    "n_snapshots": количество_расчетов,
    "doe_type": "LHS/MEPE",
    "input_scaler": "[массив_тип_скейлера_для_кажого_параметра_или_null_для_SVD]",
    "output_scaler": "[массив_тип_скейлера_для_кажого_параметра_или_null_для_SVD]"
  },
  "rom_settings": {
    "rom_type": "SVD/NN/GPR"
  },
  "variables": {
    "input_variables": [...],
    "output_variables": [...]
  }
}
```

### Поля для input_variables:
```json
{
  "name": "имя_параметра", - обчыно имеется ввиду для переменных в задаче, чаще всего радиусы и тд, как только какие-то константы или зависящие переменные, то их не учитывать
  "bounds": [мин, макс],
  "script_name": "имя_файла_макроса",
  "symbol": "символ_разделитель",
  "line": номер_строки в входном файле,
  "position": позиция_в_строке указывать по умолчанию 2
}
```

### Поля для output_variables:
- **Для SVD**: только "name"
- **Для NN/GPR**: "name указывается в ковычках в скрипте", "symbol по умолчанию пробел", "line - номер вызодной переменной", "position по умолчанию 2"

### Дополнительные настройки по методам:

#### SVD:
```json
"rom_settings": {
  "rom_type": "SVD",
  "rank": 100,
  "n_oversampling": 0,
  "n_power_iterations": 3,
  "reduction": число_или_auto
},
"postprocessor": {
  "bat_path": "путь_к_META",
  "geometry_path": "путь_к_cdb_файлу"
}
```

#### NN:
```json
"rom_settings": {
  "rom_type": "NN",
  "train_method": "lm/sgd",
  "n_epochs": 100,
  "mu": 0.001,
  "learning_rate": 0.001,
  "use_auto_search": true/false,
  "units": [массив_нейронов],
  "activations": ["tanh", "relu", "sigmoid"],
  "complexity": "Simple/Medium/Complex"
}
```

#### GPR:
```json
"rom_settings": {
  "rom_type": "GPR"
}
```

## Типы скейлеров:
- "None" - без масштабирования
- "StandardScaler" - стандартизация
- "MinMaxScaler_Symmetric" - нормализация [-1, 1]
- "MinMaxScaler_Positive" - нормализация [0, 1]
- "MinMaxScaler_Negative" - нормализация [-1, 0]
- "RobustScaler" - устойчивость к выбросам
- "PowerTransformer" - к нормальному распределению
- "LogScaler" - логарифмическое масштабирование
## ПРИМЕР ДЛЯ PREDICT конфига:

{
  "module": "ROM",
  "project": {
    "project_path": "C://Users//mai//Desktop//WorkDirectory//kirsch_problem",
    "rom_path": "C://Users//mai//Desktop//WorkDirectory//kirsch_problem//name.joblib" - Необходимо найти в папке проекта если PREDICT мод
  },
  "experiment": {
    "mode": "PREDICT",
    "needed_params": [
      [
        0.15,
        0.3
      ],
      [
        0.25,
        0.19
      ]
    ],
    "date": false
  },
  "rom_settings": {
    "rom_type": "NN"
  },
  "variables": {
    "output_variables": [
      {
        "name": "smin"
      },
      {
        "name": "smax"
      }
    ]
  }
}

При ответе:
1. Задавай уточняющие вопросы, если информации недостаточно
2. Предупреждай о важных особенностях
3. Предоставляй готовый к использованию JSON
4. КАК ТОЛЬКО СМОЖЕШЬ ПРЕДОСТАВИТЬ ГОТОВЫЙ JSON МОЖЕШЬ ВЫДАТЬ ПРОСТО JSON БЕЗ ВСЕГО ОСТАЛЬНОГО

Начинай работу с вопроса о типе задачи пользователя и постепенно собирай всю необходимую информацию.
"""
        self.initial_message = None
        self.llm = ChatMistralAI(
            model="mistral-medium-2505",
            api_key=api_key,
            temperature=0.01
        )

        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        self.chroma_client = Chroma(collection_name="test_diploma",
                                    embedding_function=HuggingFaceEmbeddings(
                                        model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={"device": "cuda"},
                                        encode_kwargs={"normalize_embeddings": True}),
                                    persist_directory="./chroma_langchain_db", )

        search_tool = Tool(
            name="search_similar_configs",
            func=self._search_similar_configs,
            description="Поиск похожих конфигураций в базе данных по описанию задачи"
        )

        reader_tool = Tool(name="read_apdl_file", func=self._read_apdl_file,
                           description="Чтение txt APDL файла для извлечения информации о переменных из файла ВСЕГДА "
                                       "ЧИТАЙ ЕСЛИ МОЖЕШЬ")

        model_finder_tool = Tool(name="find_trained_model_name", func=self.search_for_ready_model,
                                 description=(
                                     "Ищет и возвращает список путей к готовым моделям (.joblib) в указанной папке "
                                     "проекта."
                                     "Используется, когда нужно найти файл готовой модели. "
                                     "Передайте абсолютный путь к директории проекта (строкой). "
                                     "Возвращает список путей к найденным .joblib-файлам в папке (без рекурсии)."
                                 )
                                 )

        self.tools = [reader_tool, search_tool, model_finder_tool]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        self.current_config = {}
        self.config_history = []
        self.uuid = str(uuid4())

    def chat(self, message: str) -> str:
        found_config = None
        try:
            if len(self.memory.chat_memory.messages) == 0:
                self.memory.chat_memory.add_message(SystemMessage(content=self.system_prompt))
                found_config = self._search_similar_configs(message)
                self.initial_message = message
            if found_config:
                rom_path = self.search_for_ready_model(json.loads(found_config.metadata['config'])
                                                       ['project']['project_path'])
                message += f"Также в системе был уже найден конфиг {found_config}, путь до обученной модели {rom_path}"
            response = self.agent.invoke({"input": message,
                                          "chat_history": self.memory.chat_memory.messages
                                          })
            new_config = extract_json_from_response(response["output"])
            if new_config:
                self.current_config = new_config
                self.config_history.append(new_config.copy())
                print(f"Конфигурация обновлена: {len(self.current_config)} полей")

            return response["output"]

        except Exception as e:
            return f"Ошибка: {e}"

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.current_config, updates)
        return self.current_config

    def _read_apdl_file(self, file_path: str) -> str:
        try:
            file = json.loads(file_path)
            file = file["file_path"]
        except ValueError:
            file = file_path
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()

        prompt = f"""```json
            ### Поля для input_variables:
            ```json
            
              "name": "имя_параметра", - обчыно имеется ввиду для переменных в задаче, чаще всего радиусы и тд, как только какие-то константы или зависящие переменные, то их не учитывать
              "bounds": [мин, макс],
              "script_name": "имя_файла_макроса указать расширение файла если есть ",
              "symbol": "символ_разделитель",
              "line": номер_строки в входном файле,
              "position": позиция_в_строке указывать по умолчанию 2
            
            ```

            ### Поля для output_variables:
            - **Для SVD**: только "name указывается в ковычках в скрипте"
            - **Для NN/GPR**: "name указывается в ковычках в скрипте", "symbol по умолчанию пробел", "line - номер вызодной переменной", "position по умолчанию 2"
            - для выходных параметров не нужно указывать script_name

            
            текст для анализа:
            {content}
        """

        result = self.llm([HumanMessage(content=prompt)])
        summary_prompt = f"""
        Определи тип задачи и укажи основные входные и выходные переменные для данного скрипта. Формат: задача [тип задачи], входные переменные: [переменные], выходные данные: [результаты].
        Если есть возможность описать что за геометрия строится опиши её.
        {content}
        """

        self.result_summary = self.llm([HumanMessage(content=summary_prompt)])
        return result.content

    def _search_similar_configs(self, query: str) -> Document:
        try:
            result = self.chroma_client.similarity_search(query=query, k=1)
            print(result)
            return result[0] if result else None
        except ValueError as e:
            print(e)

    def get_current_config(self) -> Dict[str, Any]:
        return self.current_config.copy()

    def update_current_config_db(self):
        try:
            self.chroma_client.add_documents(ids=[self.uuid],
                                         documents=[Document(page_content=self.result_summary.content,
                                                             metadata={"config": json.dumps(self.current_config)})])
        except AttributeError:
            print("No summary needed")

    def search_for_ready_model(self, folder_path: str) -> list:
        joblib_files = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.endswith('.joblib'):
                joblib_files.append(full_path)
        return joblib_files


class ROMAgent:
    def __init__(self, api_key: str):
        self.llm = ChatMistralAI(
            model="mistral-medium-latest",
            api_key=api_key,
            temperature=0.1
        )

        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        rom_tool = Tool(
            name="run_rom_module",
            func=self._run_rom_module,
            description="Запуск ROM модуля с файлом конфигурации"
        )
        results_tool = Tool(
            name="results_reader",
            func=self._read_results_files,
            description="Чтение результатов ROM модуля только ПОСЛЕ ВЫПОЛНЕНИЯ ROM модуля"
        )
        self.tools = [rom_tool, results_tool]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        self.last_execution_path = None

    def _run_rom_module(self, json_config):
        try:
            from src.rom_llm.example import main as rom_main
            rom_main(json_config, direct_json=True)
            json_config = load(open(json_config))
            results_path = json_config["project"]["project_path"]
            return results_path
        except Exception as e:
            return f"Ошибка при запуске ROM модуля: {e}"

    def _read_results_files(self, results_directory_path: str, initial_message: str) -> str:
        try:
            result_patterns = ['results_*', 'result.txt', 'output.txt']
            result_files = []
            directory_path = results_directory_path.strip().strip('"\'')

            for pattern in result_patterns:
                files = glob.glob(os.path.join(directory_path, pattern))
                result_files.extend(files)

            if not result_files:
                return "Файлы результатов не найдены"
            result_file = result_files[0]
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            analysis_context = f"""
            АНАЛИЗ РЕЗУЛЬТАТОВ ROM МОДУЛЯ:
            Пользовательский запрос: {initial_message}
            Файл результатов: {result_file}
            Содержимое файла:
            {content}

            Это результаты работы ROM (Reduced Order Model) модуля.
            Проанализируйте следующие аспекты:
            1. Успешность выполнения расчета
            2. Полученные значения выходных параметров
            3. Возможные проблемы или аномалии
            4. Рекомендации по интерпретации результатов

            Содержимое файла для анализа:
            {content}
            """
            response = self.llm([HumanMessage(content=analysis_context)])
            return response.content


        except Exception as e:
            return f"Ошибка при чтении результатов: {e}"

    def chat(self, message: str) -> str:
        try:
            response = self.agent.invoke({
                "input": message,
                "chat_history": self.memory.chat_memory.messages
            })
            return response["output"]
        except Exception as e:
            return f"Ошибка в ROM агенте: {e}"


class AgentMode(Enum):
    CONFIG_CREATION = "config_creation"
    ROM_EXECUTION = "rom_execution"
    AUTO = "auto"


class ROMMultiAgentSystem:
    """Главный оркестратор для управления агентами ROM"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.config_agent = ROMConfigChatAgent(api_key)
        self.rom_agent = ROMAgent(api_key)
        self.current_mode = AgentMode.CONFIG_CREATION
        self.conversation_memory = ConversationBufferMemory(return_messages=True)

        self.coordinator_llm = ChatMistralAI(
            model="mistral-medium-latest",
            api_key=api_key,
            temperature=0.1
        )

        self.coordinator_prompt = """
        Ты - координатор системы ROM агентов. Твоя задача определить, какой агент должен обработать запрос пользователя.

        Доступные режимы:
        1. CONFIG_CREATION - создание/редактирование и поиск похожих ROM моделей JSON конфигурации
        2. ROM_EXECUTION - запуск ROM модуля и анализ результатов

        Критерии для CONFIG_CREATION:
        - Пользователь хочет создать новую конфигурацию
        - Нужно изменить параметры конфигурации
        - Вопросы о структуре JSON
        - Настройка переменных, методов (SVD/NN/GPR)

        Критерии для ROM_EXECUTION:
        - Пользователь хочет запустить модель
        - Готов анализировать результаты
        - Есть готовая конфигурация и нужно ее выполнить

        Отвечай одним словом: CONFIG_CREATION или ROM_EXECUTION
        """

    def determine_mode(self, user_message: str, context: str = "") -> AgentMode:
        """Определение подходящего режима на основе сообщения пользователя"""

        has_config = bool(self.config_agent.current_config)

        try:
            prompt = f"""
               {self.coordinator_prompt}

               Контекст: {"Есть готовая конфигурация" if has_config else "Нет готовой конфигурации"}
               Сообщение пользователя: {user_message}
               """

            response = self.coordinator_llm.invoke([HumanMessage(content=prompt)])

            if "ROM_EXECUTION" in response.content.upper():
                return AgentMode.ROM_EXECUTION
            else:
                return AgentMode.CONFIG_CREATION

        except Exception:
            return AgentMode.CONFIG_CREATION if not has_config else AgentMode.ROM_EXECUTION

    def chat(self, message: str, force_mode: Optional[AgentMode] = None) -> str:
        """Главный метод для общения с системой"""

        if force_mode:
            mode = force_mode
        else:
            mode = self.determine_mode(message)

        self.current_mode = mode

        self.conversation_memory.chat_memory.add_user_message(message)

        try:
            if mode == AgentMode.CONFIG_CREATION:
                response = self._handle_config_mode(message)
            elif mode == AgentMode.ROM_EXECUTION:
                response = self._handle_execution_mode(message)
            else:
                response = "Ошибка определения режима"

            self.conversation_memory.chat_memory.add_ai_message(response)

            mode_info = f"\n\n[Режим: {mode.value}]"
            return response + mode_info

        except Exception as e:
            error_msg = f"Ошибка в системе агентов: {e}"
            self.conversation_memory.chat_memory.add_ai_message(error_msg)
            return error_msg

    def _handle_config_mode(self, message: str) -> str:
        """Обработка режима создания конфигурации"""

        if any(keyword in message.lower() for keyword in ["запустить", "выполнить", "готово"]):
            if self.config_agent.current_config:
                suggestion = "\n\nКонфигурация готова! Хотите переключиться в режим выполнения? Напишите 'запустить модель' или 'выполнить'"
            else:
                suggestion = "\n\nДля запуска нужна готовая конфигурация. Продолжим настройку?"
        else:
            suggestion = ""

        response = self.config_agent.chat(message)
        return response + suggestion

    def _handle_execution_mode(self, message: str) -> str:
        """Обработка режима выполнения ROM"""

        if not self.config_agent.current_config:
            return "Для выполнения ROM модуля нужна готовая конфигурация. Переключаюсь в режим создания конфигурации."

        if any(keyword in message.lower() for keyword in ["запустить", "выполнить", "автозапуск"]):
            config_json = json.dumps(self.config_agent.current_config, ensure_ascii=False, indent=2)
            # execution_message = f"Запускаю ROM модуль с конфигурацией:\n{config_json}"
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
                temp_file_path = temp_file.name
                json.dump(self.config_agent.current_config, temp_file, ensure_ascii=False, indent=2)

            return self.rom_agent._run_rom_module(temp_file_path)

        return self.rom_agent.chat(message)

    def switch_mode(self, new_mode: AgentMode) -> str:
        """Ручное переключение режима"""
        self.current_mode = new_mode

        if new_mode == AgentMode.CONFIG_CREATION:
            return "Переключено в режим создания конфигурации. Давайте настроим параметры ROM модуля."
        elif new_mode == AgentMode.ROM_EXECUTION:
            if self.config_agent.current_config:
                return "Переключено в режим выполнения ROM. Готов запустить модель и проанализировать результаты."
            else:
                return "Для режима выполнения нужна готовая конфигурация. Сначала создайте конфигурацию."

        return f"Переключено в режим: {new_mode.value}"

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        return {
            "current_mode": self.current_mode.value,
            "has_config": bool(self.config_agent.current_config),
            "config_history_length": len(self.config_agent.config_history),
            "conversation_length": len(self.conversation_memory.chat_memory.messages)
        }

    def get_current_config(self) -> Dict[str, Any]:
        """Получение текущей конфигурации"""
        return self.config_agent.get_current_config()

    def save_config(self, file_path: str) -> str:
        """Сохранение конфигурации в файл"""
        try:
            config = self.config_agent.get_current_config()
            if not config:
                return "Нет конфигурации для сохранения"

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            return f"Конфигурация сохранена в: {file_path}"
        except Exception as e:
            return f"Ошибка сохранения: {e}"

    def load_config(self, file_path: str) -> str:
        """Загрузка конфигурации из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.config_agent.current_config = config
            self.config_agent.config_history.append(config.copy())

            return f"Конфигурация загружена из: {file_path}"
        except Exception as e:
            return f"Ошибка загрузки: {e}"


def example_usage():
    """Пример использования мульти-агентной системы"""
    api_key = "ovANor9e39OtOsGWBo6k2jINE1eQ9cUf"
    # api_key = "hAqbGrFM1aAbs6RPbE3S2IRfUuqmHdiM"

    system = ROMMultiAgentSystem(api_key)

    print("=== ROM Multi-Agent System ===")
    print("Команды:")
    print("- 'режим конфиг' - переключиться в режим создания конфигурации")
    print("- 'режим выполнение' - переключиться в режим выполнения ROM")
    print("- 'статус' - показать статус системы")
    print("- 'сохранить config.json' - сохранить конфигурацию")
    print("- 'загрузить config.json' - загрузить конфигурацию")
    print("- 'выход' - завершить работу")
    print()

    while True:
        user_input = input("Вы: ").strip()

        if user_input.lower() == 'выход':
            break
        elif user_input.lower() == 'статус':
            status = system.get_status()
            print(f"Статус: {status}")
            continue
        elif user_input.startswith('режим '):
            mode_name = user_input[6:].strip()
            if mode_name == 'конфиг':
                response = system.switch_mode(AgentMode.CONFIG_CREATION)
            elif mode_name == 'выполнение':
                response = system.switch_mode(AgentMode.ROM_EXECUTION)
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
                    temp_file_path = temp_file.name
                    json.dump(system.config_agent.current_config, temp_file, ensure_ascii=False, indent=2)
                system.config_agent.update_current_config_db()
                response = system.rom_agent._run_rom_module(temp_file_path)
                final_response = system.rom_agent._read_results_files(response, system.config_agent.initial_message)
                print(final_response)
            else:
                response = "Неизвестный режим. Доступны: 'конфиг', 'выполнение'"
        elif user_input.startswith('сохранить '):
            file_path = user_input[10:].strip()
            response = system.save_config(file_path)
        elif user_input.startswith('загрузить '):
            file_path = user_input[10:].strip()
            response = system.load_config(file_path)
        else:
            response = system.chat(user_input)

        print(f"Система: {response}")
        print()


if __name__ == "__main__":
    example_usage()

    chroma_client = Chroma(collection_name="test_diploma",
                           embedding_function=HuggingFaceEmbeddings(
                               model_name="intfloat/multilingual-e5-small",
                               model_kwargs={"device": "cuda"},
                               encode_kwargs={"normalize_embeddings": True}),
                           persist_directory="./chroma_langchain_db", )
    docs = chroma_client.get()
    print(docs)
    # def _read_results_files(results_directory_path: str) -> str:
    #     try:
    #         result_patterns = ['results_*', 'result.txt', 'output.txt']
    #         result_files = []
    #         directory_path = results_directory_path.strip().strip('"\'')
    #
    #         for pattern in result_patterns:
    #             files = glob.glob(os.path.join(directory_path, pattern))
    #             result_files.extend(files)
    #
    #         if not result_files:
    #             return "Файлы результатов не найдены"
    #         print(result_files)
    #         result_file = result_files[0]
    #         with open(result_file, 'r', encoding='utf-8') as f:
    #             content = f.read().strip()
    #         print(content)
    #         return content
    #     except Exception as e:
    #         print("error")
    #
    # _read_results_files("E:\\llm-rom\\src\\rom_llm")
# помоги мне составить конфиг для обучения GPR на 11 точках, LHS путь до apdl_скрипта: "E:\llm-rom\src\rom_llm\input_ellipse.txt" bat_path "E:\\llm-rom\\src\\rom_llm\\test.bat"  det_data ture date true скейлеры стандартные для всех переменных количество ядер: 1
# какие значения для задачи кирша при r_a и r_b 0.1 и 0.4
