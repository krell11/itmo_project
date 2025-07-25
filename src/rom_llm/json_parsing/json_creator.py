import copy
import json
import os
import re
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Literal

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.chains import LLMChain, TransformChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, ValidationError, Field, field_validator
from .classes import VariableConfig, ProjectConfig, ExperimentConfig, PostProcessorConfig, SVDConfig, NNConfig, GPRConfig
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv("E:\\llm-rom\\.env")

API_BASE = os.getenv('API')
MODEL_NAME = os.getenv("MODEL")


class JSONOutputParser(BaseOutputParser):
    """
    Улучшенный парсер JSON-вывода с расширенной обработкой ошибок и
    исправлением распространенных проблем форматирования.
    """

    def parse(self, text: str) -> Dict:
        raw_text = text
        cleaned_text = None

        logging.debug(f"Исходный текст для парсинга JSON:\n{raw_text}")

        try:

            if "```json" in text:
                parts = text.split("```json")
                if len(parts) > 1:
                    cleaned_text = parts[1].split("```")[0].strip()
            elif "```" in text:
                parts = text.split("```")
                if len(parts) > 1:
                    cleaned_text = parts[1].strip()
            else:

                first_brace = text.find('{')
                last_brace = text.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    cleaned_text = text[first_brace:last_brace + 1]
                else:
                    cleaned_text = text.strip()

            if not cleaned_text:
                logging.warning("Не удалось выделить JSON из текста")
                cleaned_text = text.strip()

            try:
                if cleaned_text.startswith('{{'):
                    brace_count = 0
                    for i, char in enumerate(cleaned_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 1 and i < len(cleaned_text) - 1:
                                cleaned_text = cleaned_text[1:i + 1]
                                logging.warning(f"Обнаружена и исправлена двойная вложенность JSON: {cleaned_text}")
                                break

                last_brace = cleaned_text.rfind('}')
                if last_brace != -1:
                    cleaned_text = cleaned_text[:last_brace + 1]

                open_braces = cleaned_text.count('{')
                close_braces = cleaned_text.count('}')
                if open_braces > close_braces:
                    cleaned_text += '}' * (open_braces - close_braces)
                    logging.warning(f"Добавлены отсутствующие закрывающие скобки: {open_braces - close_braces}")
                elif close_braces > open_braces:
                    first_brace = cleaned_text.find('{')
                    cleaned_text = cleaned_text[first_brace:]
                    open_braces = cleaned_text.count('{')
                    close_braces = cleaned_text.count('}')
                    if close_braces > open_braces:
                        cleaned_text = cleaned_text[:-(close_braces - open_braces)]
                    logging.warning(f"Удалены лишние закрывающие скобки: {close_braces - open_braces}")

                cleaned_text = re.sub(r',\s*}', '}', cleaned_text)
                cleaned_text = re.sub(r',\s*]', ']',
                                      cleaned_text)
                python_code_match = re.search(r'}\s*\n\s*#', cleaned_text)
                if python_code_match:
                    cleaned_text = cleaned_text[:python_code_match.start() + 1]
                    logging.warning("Обнаружен и удален Python-код после JSON")

            except Exception as e:
                logging.warning(f"Ошибка при исправлении формата JSON: {str(e)}")

            logging.debug(f"Очищенный текст для парсинга JSON:\n{cleaned_text}")

            try:
                parsed = json.loads(cleaned_text)
                logging.debug(f"Успешно распарсен JSON:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}")
                return parsed
            except json.JSONDecodeError as e:
                pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
                matches = re.findall(pattern, cleaned_text)
                if matches:
                    for potential_json in matches:
                        try:
                            parsed = json.loads(potential_json)
                            logging.warning(f"Успешно извлечен JSON с помощью регулярного выражения")
                            return parsed
                        except:
                            continue

                raise e

        except json.JSONDecodeError as e:
            error_pos = max(0, e.pos - 20)
            error_end = min(len(cleaned_text), e.pos + 20) if cleaned_text else 0
            error_context = cleaned_text[error_pos:error_end] if cleaned_text else ""

            marked_context = (
                    error_context[:e.pos - error_pos] +
                    " ---> HERE ---> " +
                    error_context[e.pos - error_pos:]
            ) if error_pos <= e.pos <= error_end else error_context

            error_msg = (
                f"Ошибка JSON: {e.msg} (позиция {e.pos})\n"
                f"Контекст ошибки с маркером: '{marked_context}'\n"
                f"Полный очищенный текст:\n{cleaned_text}\n"
                f"Исходный текст:\n{raw_text[0:500]}..." if len(raw_text) > 500 else raw_text
            )
            logging.error(error_msg)
            raise OutputParserException(error_msg)

        except Exception as e:
            error_msg = f"Неизвестная ошибка при парсинге JSON: {str(e)}\n"
            if cleaned_text:
                error_msg += f"Очищенный текст:\n{cleaned_text}\n"
            error_msg += f"Исходный текст:\n{raw_text[0:500]}..." if len(raw_text) > 500 else raw_text
            logging.error(error_msg)
            raise OutputParserException(error_msg)


def correct_bounds_in_config(user_input: str, config: Dict) -> Dict:
    """
    Выполняет коррекцию границ переменных в конфигурации на основе пользовательского запроса.

    Args:
        user_input: Исходный пользовательский запрос
        config: Сгенерированная конфигурация

    Returns:
        Обновленная конфигурация с исправленными границами
    """
    corrected_config = copy.deepcopy(config)

    input_vars = []
    for var in corrected_config.get("variables", {}).get("input_variables", []):
        if "name" in var:
            input_vars.append(var["name"])

    if not input_vars:
        logging.warning("В конфигурации не найдены входные переменные для коррекции границ")
        return corrected_config

    llm = ChatOpenAI(
        base_url=API_BASE,
        model_name=MODEL_NAME,
        openai_api_key="dummy_key",
        temperature=0,
        max_tokens=524
    )

    prompt_template = PromptTemplate(
        input_variables=["user_input", "config_json", "input_vars"],
        template="""
        ЗАДАЧА: Проанализируй запрос пользователя и исправь границы (bounds) для переменных в конфигурации.

        Запрос пользователя:
        ---
        {user_input}
        ---

        Входные переменные: {input_vars}

        ИНСТРУКЦИИ:
        1. Внимательно изучи запрос пользователя на наличие информации о ГРАНИЦАХ параметров
        2. Найди упоминания границ в формах:
           - "параметры в границах (диапазоне) X до Y"
           - "переменная VAR от X до Y"
           - "диапазон значений X-Y"
           - "границы [X, Y]"
           - "min=X, max=Y"
        3. Если в запросе указаны границы для конкретных переменных, используй их для соответствующих переменных
        4. Если в запросе указаны общие границы без привязки к конкретным переменным, примени их ко ВСЕМ переменным
        5. Если указаны только минимальное ИЛИ только максимальное значение, дополни недостающее значение: 
           - если указан только минимум, установи максимум = минимум * 10
           - если указан только максимум, установи минимум = максимум / 10
        6. Не трогай значения, которые не указаны в запросе, если для них уже заданы границы
        7. Если ни для одной переменной не удается установить границы на основе запроса, установи [0.1, 1.0] для всех переменных

        ВЕРНИ ТОЛЬКО СЛОВАРЬ С ИСПРАВЛЕННЫМИ ГРАНИЦАМИ:
        {{
            "имя_переменной1": [мин, макс],
            "имя_переменной2": [мин, макс],
            ...
        }}

        Важно: верни только JSON словарь с границами, без пояснений и других данных!
        """
    )

    try:
        config_json = json.dumps(corrected_config, indent=2, ensure_ascii=False)
        input_vars_str = ", ".join(input_vars)

        bounds_chain = (
                prompt_template
                | llm
                | StrOutputParser()
                | JSONOutputParser()
        )

        corrected_bounds = bounds_chain.invoke({
            "user_input": user_input,
            "config_json": config_json,
            "input_vars": input_vars_str
        })

        logging.info(f"Обнаружены и исправлены границы: {corrected_bounds}")

        if corrected_bounds and isinstance(corrected_bounds, dict):
            for var in corrected_config.get("variables", {}).get("input_variables", []):
                var_name = var.get("name")
                if var_name in corrected_bounds and isinstance(corrected_bounds[var_name], list):
                    var["bounds"] = corrected_bounds[var_name]
                    logging.info(f"Установлены исправленные границы для {var_name}: {var['bounds']}")

            if "variables" in corrected_bounds and "input_variables" in corrected_bounds.get("variables", {}):
                input_vars_data = corrected_bounds["variables"]["input_variables"]
                if isinstance(input_vars_data, list):
                    for input_var in input_vars_data:
                        var_name = input_var.get("name")
                        bounds = input_var.get("bounds")
                        if var_name and bounds:
                            for var in corrected_config.get("variables", {}).get("input_variables", []):
                                if var.get("name") == var_name:
                                    var["bounds"] = bounds
                                    logging.info(f"Установлены исправленные границы для {var_name}: {bounds}")
                                    break

        return corrected_config

    except Exception as e:
        logging.error(f"Ошибка при коррекции границ переменных: {str(e)}")
        return corrected_config


class ConfigTools:
    """
    Класс с инструментами для работы с конфигурацией ROM-модуля
    """

    @staticmethod
    def parse_script(
            file_path: str,
            target_input_vars: Optional[List[str]] = None,
            target_output_vars: Optional[List[str]] = None
    ) -> Dict:
        """
        Инструмент 1: Анализ APDL скрипта для извлечения информации о переменных.

        Args:
            file_path: Путь к файлу скрипта
            target_input_vars: Опциональный список конкретных входных переменных для поиска
            target_output_vars: Опциональный список конкретных выходных переменных для поиска

        Returns:
            Словарь с информацией о входных и выходных переменных
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл скрипта не найден: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except Exception as e:
            raise IOError(f"Ошибка при чтении файла {file_path}: {str(e)}")

        if target_input_vars is None:
            target_input_vars_str = "r_a строка 4, r_b строка 5"
        else:
            target_input_vars_str = ", ".join([f"{var}" for var in target_input_vars])

        if target_output_vars is None:
            target_output_vars_str = "smin, smax"
        else:
            target_output_vars_str = ", ".join(target_output_vars)

        llm = ChatOpenAI(
            base_url=API_BASE,
            model_name=MODEL_NAME,
            openai_api_key="dummy_key",
            temperature=0,
            max_tokens=1000
        )

        prompt_template = PromptTemplate(
            input_variables=["script_apdl", "target_input_vars", "target_output_vars"],
            template="""
               Проанализируй APDL скрипт и извлеки из него ТОЛЬКО параметры {target_input_vars} (входные) и {target_output_vars} (выходные).

               ВЕРНИ СТРОГО ТОЛЬКО JSON-ОБЪЕКТ БЕЗ ПОЯСНЕНИЙ В СЛЕДУЮЩЕМ ФОРМАТЕ:
               {{
                 "input_variables": [
                   {{
                     "name": "имя переменной",
                     "bounds": [],
                     "script_name": "имя файла",
                     "symbol": "=",
                     "line": номер_строки во всодном файле,
                     "position": 2
                   }}
                 ],
                 "output_variables": [
                   {{
                     "name": "имя переменной",
                     "symbol": " ",
                     "line": номер_строки в выходном файле,
                     "position": 2
                   }}
                 ]
               }}

               ВНИМАНИЕ: 
               1. НЕ ДОБАВЛЯЙ НИКАКИХ ПОЯСНЕНИЙ И ВВОДНОГО ТЕКСТА
               2. НЕ ИСПОЛЬЗУЙ MARKDOWN ФОРМАТИРОВАНИЕ (```json)
               3. ВЕРНИ ТОЛЬКО ГОЛЫЙ JSON-ОБЪЕКТ
               4. Найди строки, где определяются входные переменные {target_input_vars} и где они используются
               5. В "name" укажи точное имя переменной как в скрипте
               6. В "symbol" укажи символ-разделитель (обычно "=" для входных и " " для выходных)
               7. В "line" укажи номер строки в тексте (начиная с 1), для output_variables этот параметр означает номер строки в выходном файле или порядок записи переменной
               8. В "position" укажи позицию после разделения по символу (обычно 2)

               APDL скрипт для анализа:
               {script_apdl}
               """
        )

        try:

            chain = (
                    prompt_template
                    | llm
                    | StrOutputParser()
                    | JSONOutputParser()
            )

            result = chain.invoke({
                "script_apdl": script_content,
                "target_input_vars": target_input_vars_str,
                "target_output_vars": target_output_vars_str
            })
            logging.debug(result)
            if not isinstance(result, dict):
                raise ValueError(f"Ожидался словарь, получено: {type(result)}")

            required_keys = ["input_variables", "output_variables"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"В результате отсутствует обязательный ключ: {key}")

            for i, var in enumerate(result.get("input_variables", [])):
                if "name" not in var:
                    raise ValueError(f"Отсутствует имя для входной переменной {i + 1}")

            for i, var in enumerate(result.get("output_variables", [])):
                if "name" not in var:
                    raise ValueError(f"Отсутствует имя для выходной переменной {i + 1}")

            return result

        except Exception as e:
            error_msg = f"Ошибка при анализе скрипта {file_path}: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def create_base_config(
            user_input: str,
            use_json_schema: bool = False,
            rom_type: Optional[str] = None,
            mode: Optional[str] = None
    ) -> Dict:
        """
        Инструмент 2: Создание базовой конфигурации на основе пользовательского ввода

        Args:
            user_input: Пользовательский ввод для генерации конфигурации
            use_json_schema: Использовать ли JSON Schema для подсказки LLM
            rom_type: Тип ROM модели (SVD, NN, GPR)
            mode: Режим работы (TRAIN, PREDICT, TP)

        Returns:
            Базовая конфигурация в виде словаря
        """

        llm = ChatOpenAI(
            base_url=API_BASE,
            model_name=MODEL_NAME,
            openai_api_key="dummy_key",
            temperature=0,

        )

        json_schema = ""
        if use_json_schema:
            project_schema = json.dumps(ProjectConfig.model_json_schema(), indent=2)
            experiment_schema = json.dumps(ExperimentConfig.model_json_schema(), indent=2)

            rom_schemas = {
                "SVD": SVDConfig.model_json_schema(),
                "NN": NNConfig.model_json_schema(),
                "GPR": GPRConfig.model_json_schema()
            }

            if rom_type and rom_type in rom_schemas:
                rom_schema = json.dumps(rom_schemas[rom_type], indent=2)
            else:
                rom_schema = json.dumps({
                    "oneOf": [
                        SVDConfig.model_json_schema(),
                        NNConfig.model_json_schema(),
                        GPRConfig.model_json_schema()
                    ]
                }, indent=2)

            variable_schema = json.dumps(VariableConfig.model_json_schema(), indent=2)

            json_schema = f"""
            JSON Schema для валидации:

            Project Schema:
            {project_schema}

            Experiment Schema:
            {experiment_schema}

            ROM Schema:
            {rom_schema}

            Variable Schema:
            {variable_schema}
            """

        mode_instruction = ""
        if mode:
            mode_instruction = f"Режим работы должен быть установлен на '{mode}'."

        rom_type_instruction = ""
        if rom_type:
            rom_type_instruction = f"Тип ROM модели должен быть установлен на '{rom_type}'."

        prompt_template = PromptTemplate(
            input_variables=["user_input", "json_schema", "mode_instruction", "rom_type_instruction"],
            template="""Извлеки параметры конфигурации в формате JSON ОБЯЗАТЕЛЬНО с полной структурой:
            {{
                "project": {{  # ОБЯЗАТЕЛЬНЫЙ РАЗДЕЛ
                    "project_path": "путь/к/проекту (обязательно)",
                    "script_paths": ["макрос1.txt", "макрос2.txt"],
                    "bat_path": "файл.bat",
                    "results_file": "результаты.txt",
                    "number_of_cores": 4,
                    "del_data": true,
                    "date": true,
                    "doe_file": "doe.npz(только если пользователь хочет дообучить модель)",
                    "project_name": "мой_проект"
                }},
                "experiment": {{  # ОБЯЗАТЕЛЬНЫЙ РАЗДЕЛ
                    "mode": "TRAIN/PREDICT/TP",
                    "n_snapshots": 100,
                    "doe_type": "LHS/MEPE(MEPE для адаптивного датасета)",
                }},
                "rom_settings": {{  # ОБЯЗАТЕЛЬНЫЙ РАЗДЕЛ
                    "rom_type": "SVD/NN/GPR (обязательно)",
                    "n_power_iters": 5(для SVD модели),
                }},
                "variables": {{  # ОБЯЗАТЕЛЬНЫЙ РАЗДЕЛ
                    "input_variables": [
                        {{
                            "name": "параметр_ввода",
                            "bounds": [0, 1],
                            "script_name": "скрипт.txt",
                            "symbol": "=",
                            "line": 5,
                            "position": 2
                        }}
                    ],
                    "output_variables": [
                        {{
                            "name": "параметр_вывода",
                            "symbol": " ",
                            "line": 10,
                            "position": 1
                        }}
                    ]
                }}
            }}

            {json_schema}

            ВНИМАНИЕ: ВСЕ ЧЕТЫРЕ РАЗДЕЛА (project, experiment, rom_settings, variables) ДОЛЖНЫ ПРИСУТСТВОВАТЬ В JSON!

            {mode_instruction}
            {rom_type_instruction}

            Запрос: {user_input}
            Ответ ТОЛЬКО JSON, НЕ НУЖНО ПИСАТЬ ЧТО-ТО КРОМЕ JSON"""
        )

        extract_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_parser=JSONOutputParser(),
            output_key="raw_config"
        )

        def build_and_validate_config(inputs: Dict) -> Dict:
            """
            Строит и валидирует полную конфигурацию на основе исходных данных.
            """
            raw_config = inputs["raw_config"]
            error_messages = None
            full_config = None
            logging.debug(raw_config)
            logging.debug(f"Исходная конфигурация:\n{json.dumps(raw_config, indent=2, ensure_ascii=False)}")

            try:

                required_sections = ["project", "experiment", "rom_settings", "variables"]
                missing_sections = [section for section in required_sections if section not in raw_config]
                if missing_sections:
                    raise ValueError(f"Отсутствуют обязательные разделы: {', '.join(missing_sections)}")

                project_config = ProjectConfig(**raw_config.get("project", {}))
                experiment_config = ExperimentConfig(**raw_config.get("experiment", {}))

                if mode:
                    experiment_config.mode = mode

                input_vars = [
                    VariableConfig(**var)
                    for var in raw_config.get("variables", {}).get("input_variables", [])
                ]
                output_vars = [
                    VariableConfig(**var)
                    for var in raw_config.get("variables", {}).get("output_variables", [])
                ]

                if not input_vars:
                    raise ValueError("Не указаны входные переменные")
                if not output_vars:
                    raise ValueError("Не указаны выходные переменные")

                full_config = {
                    "module": "ROM",
                    "project": project_config.model_dump(exclude_none=True),
                    "experiment": experiment_config.model_dump(exclude_none=True),
                    "variables": {
                        "input_variables": [var.model_dump(exclude_none=True) for var in input_vars],
                        "output_variables": [var.model_dump(exclude_none=True) for var in output_vars]
                    }
                }

                rom_type_actual = rom_type if rom_type else raw_config.get("rom_settings", {}).get("rom_type", "SVD")

                if rom_type_actual == "SVD":
                    full_config["rom_settings"] = SVDConfig(
                        **raw_config.get("rom_settings", {})
                    ).model_dump(exclude_none=True)

                    if "postprocessor" in raw_config:
                        full_config["postprocessor"] = PostProcessorConfig(
                            **raw_config.get("postprocessor", {})
                        ).model_dump(exclude_none=True)

                elif rom_type_actual == "NN":
                    full_config["rom_settings"] = NNConfig(
                        **raw_config.get("rom_settings", {})
                    ).model_dump(exclude_none=True)
                elif rom_type_actual == "GPR":
                    full_config["rom_settings"] = GPRConfig(
                        **raw_config.get("rom_settings", {})
                    ).model_dump(exclude_none=True)
                else:
                    raise ValueError(f"Неподдерживаемый тип ROM модели: {rom_type_actual}")

                logging.info("Конфигурация успешно валидирована")

                return {
                    "unvalidated_config": full_config,
                    "validation_errors": None
                }

            except ValidationError as e:
                error_messages = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
                logging.error(f"Ошибки валидации:\n{json.dumps(error_messages, indent=2, ensure_ascii=False)}")

                return {
                    "unvalidated_config": None,
                    "validation_errors": error_messages
                }
            except Exception as e:
                error_message = str(e)
                logging.error(f"Ошибка при валидации конфигурации: {error_message}")

                return {
                    "unvalidated_config": None,
                    "validation_errors": [error_message]
                }

        def handle_validation(inputs: Dict) -> Dict:
            """
            Обрабатывает результаты валидации.
            """
            if inputs.get("validation_errors"):
                formatted_errors = "\n".join([f"- {err}" for err in inputs["validation_errors"]])
                raise ValueError(f"Ошибки валидации конфигурации:\n{formatted_errors}")

            return {
                "full_config": inputs["unvalidated_config"]
            }

        validation_chain = TransformChain(
            input_variables=["raw_config"],
            output_variables=["unvalidated_config", "validation_errors"],
            transform=build_and_validate_config
        )

        error_handler_chain = TransformChain(
            input_variables=["unvalidated_config", "validation_errors"],
            output_variables=["full_config"],
            transform=handle_validation
        )

        full_chain = SequentialChain(
            chains=[extract_chain, validation_chain, error_handler_chain],
            input_variables=["user_input", "json_schema", "mode_instruction", "rom_type_instruction"],
            output_variables=["full_config"],
            verbose=True
        )

        result = full_chain.invoke({
            "user_input": user_input,
            "json_schema": json_schema,
            "mode_instruction": mode_instruction,
            "rom_type_instruction": rom_type_instruction
        })

        return result["full_config"]


class ConfigFactory:
    """
    Фабрика для создания и управления конфигурациями ROM-модуля
    с извлечением большинства параметров из текстового запроса
    """

    @staticmethod
    def create_config(
            user_input: str,
            script_path: str,
            mode: Optional[str] = None,
            bat_path: Optional[str] = None,
            results_file: Optional[str] = None,
            project_path: Optional[str] = None,
            model_path: Optional[str] = None,
            doe_file: Optional[str] = None,
            postprocessor: Optional[Dict[str, Any]] = None,
            use_json_schema: bool = False,
            target_vars: Optional[Dict[str, List[str]]] = None
    ) -> Dict:
        """
        Универсальный метод создания конфигурации, который извлекает параметры
        из пользовательского запроса и выбирает режим работы автоматически

        Args:
            user_input: Пользовательский ввод с указанием всех параметров
            script_path: Путь к APDL скрипту
            mode: Явное указание режима (опционально, иначе определяется из промпта)
            bat_path: Путь к bat файлу
            results_file: Имя файла результатов
            project_path: Путь к проекту
            model_path: Путь к обученной модели (для PREDICT)
            doe_file: Путь к файлу с существующим DOE
            postprocessor: Настройки постпроцессора (для SVD)
            use_json_schema: Использовать ли JSON Schema для подсказки LLM
            target_vars: Словарь с целевыми переменными для парсинга скрипта

        Returns:
            Конфигурация в виде словаря
        """

        llm = ChatOpenAI(
            base_url=API_BASE,
            model_name=MODEL_NAME,
            openai_api_key="dummy_key",
            temperature=0,
        )

        extract_params_template = PromptTemplate(
            input_variables=["user_input"],
            template="""
            Из запроса пользователя извлеки все параметры конфигурации для ROM-модуля.
            обязательно укажи rom_type, mode
            Верни результат в формате JSON:
            {{
                "mode": "TRAIN, PREDICT или TP",
                "rom_type": "SVD, NN или GPR",
                "n_snapshots": число (если применимо, НЕ ПРОПУСКАЙ, внимательно ищи число образцов/снапшотов в запросе),
                "doe_type": "LHS или MEPE (если применимо, ищи упоминания о методе генерации/выборки/адаптивном методе)",
                "number_of_cores": число (число ядер/процессоров/потоков),
                "needed_params": [массив с точками для предсказания, если применимо],
                "other_params": {{
                    другие параметры специфичные для типа ROM
                    }}
            }}

            Запрос: {user_input}

            Ответ только в JSON, без пояснений:
            """
        )

        extract_params_chain = (
                extract_params_template
                | llm
                | StrOutputParser()
                | JSONOutputParser()
        )

        try:
            extracted_params = extract_params_chain.invoke({"user_input": user_input})
            config_mode = mode if mode else extracted_params.get("mode")
            if not config_mode:
                if "предсказ" in user_input.lower() and "обуч" in user_input.lower():
                    config_mode = "TP"
                elif "предсказ" in user_input.lower():
                    config_mode = "PREDICT"
                else:
                    config_mode = "TRAIN"

            rom_type = extracted_params.get("rom_type")

            n_snapshots = extracted_params.get("n_snapshots")
            doe_type = extracted_params.get("doe_type")

            needed_params = extracted_params.get("needed_params")

            number_of_cores = extracted_params.get("number_of_cores")

            other_params = extracted_params.get("other_params", {})

            logging.info(f"Извлеченные параметры: mode={config_mode}, rom_type={rom_type}, "
                         f"n_snapshots={n_snapshots}, doe_type={doe_type}")

        except Exception as e:
            logging.warning(f"Ошибка извлечения параметров из запроса: {str(e)}")
            config_mode = mode if mode else "TRAIN"
            rom_type = None
            n_snapshots = 100
            doe_type = "LHS"
            needed_params = None
            number_of_cores = 4
            other_params = {}

        base_config = ConfigTools.create_base_config(
            user_input=user_input,
            use_json_schema=use_json_schema,
            rom_type=rom_type,
            mode=config_mode
        )

        target_input_vars = None
        target_output_vars = None
        if target_vars:
            target_input_vars = target_vars.get("input")
            target_output_vars = target_vars.get("output")

        variables = ConfigTools.parse_script(
            file_path=script_path,
            target_input_vars=target_input_vars,
            target_output_vars=target_output_vars
        )

        result_config = copy.deepcopy(base_config)
        result_config["variables"] = variables

        result_config["experiment"]["mode"] = config_mode

        if project_path:
            result_config["project"]["project_path"] = project_path

        if bat_path:
            result_config["project"]["bat_path"] = bat_path

        if results_file:
            result_config["project"]["results_file"] = results_file

        result_config["project"]["number_of_cores"] = number_of_cores

        result_config["project"]["script_paths"] = [script_path]
        for var in result_config["variables"]["input_variables"]:
            if "script_name" in var:
                var["script_name"] = os.path.basename(script_path)

        if config_mode == "TRAIN":
            result_config["experiment"]["n_snapshots"] = n_snapshots
            result_config["experiment"]["doe_type"] = doe_type

            if doe_file:
                result_config["project"]["doe_file"] = doe_file
            else:
                result_config["project"].pop("doe_file", None)
            input_vars_count = len(result_config.get("variables", {}).get("input_variables", []))
            output_vars_count = len(result_config.get("variables", {}).get("output_variables", []))

            if input_vars_count > 0:
                result_config["experiment"]["input_scaler"] = ["StandardScaler"] * input_vars_count

            if output_vars_count > 0:
                result_config["experiment"]["output_scaler"] = ["StandardScaler"] * output_vars_count
        elif config_mode == "PREDICT":

            result_config["experiment"].pop("n_snapshots", None)
            result_config["experiment"].pop("doe_type", None)
            result_config["project"].pop("number_of_cores", None)
            result_config["project"].pop("script_paths", None)
            result_config["project"].pop("bat_path", None)
            result_config["project"].pop("result_file", None)
            result_config["project"].pop("del_data", None)
            result_config["project"].pop("doe_file", None)
            if model_path:
                result_config["project"]["rom_path"] = model_path

            result_config["variables"].pop("input_variables", None)
            new_rom_settings = result_config["rom_settings"]["rom_type"]
            result_config["rom_settings"] = new_rom_settings
            new_output_vars = []
            for var in result_config["variables"]["output_variables"]:
                new_output_vars.append({"name": var["name"]})

            result_config["variables"]["output_variables"] = new_output_vars

            if needed_params:
                result_config["experiment"]["needed_params"] = needed_params

        elif config_mode == "TP":
            result_config["experiment"]["n_snapshots"] = n_snapshots
            result_config["experiment"]["doe_type"] = doe_type

            if doe_file:
                result_config["project"]["doe_file"] = doe_file
            else:
                result_config["project"].pop("doe_file", None)

            if needed_params:
                result_config["experiment"]["needed_params"] = needed_params

        if postprocessor and result_config.get("rom_settings", {}).get("rom_type") == "SVD":
            result_config["postprocessor"] = postprocessor

        if other_params:
            rom_settings = result_config.get("rom_settings", {})
            if rom_settings.get("rom_type") == "NN":
                if "n_epochs" in other_params:
                    rom_settings["n_epochs"] = other_params["n_epochs"]
                if "train_method" in other_params:
                    rom_settings["train_method"] = other_params["train_method"]
                if "learning_rate" in other_params:
                    rom_settings["learning_rate"] = other_params["learning_rate"]
                if "mu" in other_params:
                    rom_settings["mu"] = other_params["mu"]
                if "use_auto_search" in other_params:
                    rom_settings["use_auto_search"] = other_params["use_auto_search"]
                if "complexity" in other_params:
                    rom_settings["complexity"] = other_params["complexity"]

            elif rom_settings.get("rom_type") == "SVD":
                if "rank" in other_params:
                    rom_settings["rank"] = other_params["rank"]
                if "n_oversampling" in other_params:
                    rom_settings["n_oversampling"] = other_params["n_oversampling"]
                if "n_power_iters" in other_params:
                    rom_settings["n_power_iters"] = other_params["n_power_iters"]
                if "reduction" in other_params:
                    rom_settings["reduction"] = other_params["reduction"]

        result_config = correct_bounds_in_config(user_input, result_config)

        return result_config

    @staticmethod
    def save_config(config: Dict, output_path: str) -> None:
        """
        Сохраняет конфигурацию в JSON файл

        Args:
            config: Конфигурация для сохранения
            output_path: Путь для сохранения файла
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('config_creation.log', encoding='utf-8')
        ]
    )


if __name__ == "__main__":
    setup_logging()
    predict_config = ConfigFactory.create_config(
        user_input="создай конфиг чтобы обучить GPR на 11 точках, doe_type пусть будет LHS, для параметров  в диапазоне [0.1, 0.5], количество ядер 1",
        script_path="E:/llm-rom/src/rom_llm/input_ellipse.txt",
        bat_path="E:/llm-rom/src/rom_llm/test.bat",
        results_file="result.txt",
        project_path="E:\\ROM",
        model_path="E:\\ROM\\gpr_model_2025_0513_1434.joblib"
    )

    ConfigFactory.save_config(predict_config, "/src/rom_llm/predict_config.json")

    print(json.dumps(predict_config, indent=2, ensure_ascii=False))
