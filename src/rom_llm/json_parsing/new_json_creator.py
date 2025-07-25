import copy
import json
import os
import re
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv("E:\\llm-rom\\.env")

API_BASE = os.getenv('API')
MODEL_NAME = os.getenv("MODEL")


class ImprovedJSONOutputParser(BaseOutputParser):
    def parse(self, text: str) -> Dict:
        original_text = text.strip()

        json_candidates = self._extract_json_candidates(original_text)

        for candidate in json_candidates:
            try:
                cleaned = self._clean_json_text(candidate)
                parsed = json.loads(cleaned)
                logging.info("JSON успешно распарсен")
                return parsed
            except json.JSONDecodeError as e:
                logging.debug(f"Ошибка парсинга кандидата: {e}")
                continue

        for candidate in json_candidates:
            try:
                fixed = self._fix_json_errors(candidate)
                parsed = json.loads(fixed)
                logging.warning("JSON распарсен после исправления ошибок")
                return parsed
            except:
                continue

        raise OutputParserException(f"Не удалось распарсить JSON из текста: {original_text[:500]}...")

    def _extract_json_candidates(self, text: str) -> List[str]:
        """Извлекает возможные JSON блоки из текста"""
        candidates = []

        if "```json" in text:
            matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            candidates.extend(matches)

        if "```" in text and not candidates:
            matches = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
            candidates.extend(matches)

        brace_matches = re.findall(r'(\{.*\})', text, re.DOTALL)
        candidates.extend(brace_matches)

        candidates.append(text)

        return candidates

    def _clean_json_text(self, text: str) -> str:
        """Базовая очистка JSON текста"""
        text = text.strip()

        text = re.sub(r'#.*?\n', '\n', text)
        text = re.sub(r'//.*?\n', '\n', text)

        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)

        return text

    def _fix_json_errors(self, text: str) -> str:
        """Исправление типичных ошибок JSON"""
        text = self._clean_json_text(text)

        open_braces = text.count('{')
        close_braces = text.count('}')

        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            excess = close_braces - open_braces
            for _ in range(excess):
                text = text.rstrip('}').rstrip()

        text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

        return text


def extract_parameters_with_fallback(user_input: str) -> Dict:

    llm = ChatOpenAI(
        base_url=API_BASE,
        model_name=MODEL_NAME,
        openai_api_key="dummy_key",
        temperature=0,
        max_tokens=800
    )

    prompt = PromptTemplate.from_template("""
Извлеки параметры для ROM-модуля из запроса. Верни ТОЛЬКО JSON:

{{
    "mode": "TRAIN или PREDICT или TP",
    "rom_type": "SVD или NN или GPR", 
    "n_snapshots": число_образцов_для_обучения,
    "doe_type": "LHS или MEPE",
    "number_of_cores": число_ядер,
    "needed_params": [[значения для предсказания]] или null
}}

Правила:
- mode: TRAIN если только обучение, PREDICT если только предсказание, TP если и то и другое
- rom_type: ищи упоминания SVD/нейрон/GPR/гаусс
- n_snapshots: ищи числа образцов/снапшотов/точек для обучения
- doe_type: MEPE если упоминается адаптивный метод, иначе LHS

Запрос: {user_input}

JSON:""")

    try:
        chain = prompt | llm | StrOutputParser() | ImprovedJSONOutputParser()
        result = chain.invoke({"user_input": user_input})

        validated = {
            "mode": result.get("mode", "TRAIN"),
            "rom_type": result.get("rom_type", "SVD"),
            "n_snapshots": result.get("n_snapshots", 100),
            "doe_type": result.get("doe_type", "LHS"),
            "number_of_cores": result.get("number_of_cores", 4),
            "needed_params": result.get("needed_params")
        }

        logging.info(f"Извлеченные параметры: {validated}")
        return validated

    except Exception as e:
        logging.warning(f"Ошибка извлечения параметров: {e}. Используем значения по умолчанию")
        return {
            "mode": "TRAIN",
            "rom_type": "SVD",
            "n_snapshots": 100,
            "doe_type": "LHS",
            "number_of_cores": 4,
            "needed_params": None
        }


def create_minimal_config(params: Dict, script_path: str, **kwargs) -> Dict:
    """
    Создание минимальной рабочей конфигурации без LLM
    """
    script_name = os.path.basename(script_path)

    config = {
        "module": "ROM",
        "project": {
            "project_path": kwargs.get("project_path", "/default/project/path"),
            "script_paths": [script_path],
            "bat_path": kwargs.get("bat_path", "/default/batch.bat"),
            "results_file": kwargs.get("results_file", "result.txt"),
            "number_of_cores": params["number_of_cores"],
            "del_data": False,
            "date": True,
            "project_name": "rom_project"
        },
        "experiment": {
            "mode": params["mode"],
            "doe_type": params["doe_type"]
        },
        "rom_settings": {
            "rom_type": params["rom_type"]
        },
        "variables": {
            "input_variables": [
                {
                    "name": "param1",
                    "bounds": [0.1, 1.0],
                    "script_name": script_name,
                    "symbol": "=",
                    "line": 1,
                    "position": 2
                }
            ],
            "output_variables": [
                {
                    "name": "result1",
                    "symbol": " ",
                    "line": 1,
                    "position": 2
                }
            ]
        }
    }

    if params["mode"] in ["TRAIN", "TP"]:
        config["experiment"]["n_snapshots"] = params["n_snapshots"]

    if params["mode"] in ["PREDICT", "TP"]:
        if params["needed_params"]:
            config["experiment"]["needed_params"] = params["needed_params"]
        if kwargs.get("model_path"):
            config["project"]["rom_path"] = kwargs["model_path"]

    if params["rom_type"] == "NN":
        config["rom_settings"].update({
            "train_method": "lm",
            "n_epochs": 100,
            "use_auto_search": False
        })
    elif params["rom_type"] == "SVD":
        config["rom_settings"].update({
            "rank": 100,
            "n_oversampling": 0,
            "n_power_iters": 3
        })

    return config


def parse_script_simple(script_path: str) -> Dict:
    if not os.path.exists(script_path):
        logging.warning(f"Скрипт не найден: {script_path}")
        return {
            "input_variables": [],
            "output_variables": []
        }

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        input_vars = []
        output_vars = []

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if '=' in line and not line.startswith('*') and not line.startswith('!'):
                parts = line.split('=')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    if var_name.replace('_', '').isalnum():
                        input_vars.append({
                            "name": var_name,
                            "bounds": [0.1, 1.0],
                            "script_name": os.path.basename(script_path),
                            "symbol": "=",
                            "line": i,
                            "position": 2
                        })

        output_vars = [
            {
                "name": "output1",
                "symbol": " ",
                "line": 1,
                "position": 2
            }
        ]

        logging.info(f"Найдено {len(input_vars)} входных переменных в скрипте")
        return {
            "input_variables": input_vars[:5],  # Ограничиваем количество
            "output_variables": output_vars
        }

    except Exception as e:
        logging.error(f"Ошибка парсинга скрипта: {e}")
        return {
            "input_variables": [],
            "output_variables": []
        }


class StableConfigFactory:

    @staticmethod
    def create_config(
            user_input: str,
            script_path: str,
            mode: Optional[str] = None,
            bat_path: Optional[str] = None,
            results_file: Optional[str] = None,
            project_path: Optional[str] = None,
            model_path: Optional[str] = None,
            use_llm_parsing: bool = True
    ) -> Dict:
        """
        Создание конфигурации с опциональным использованием LLM

        Args:
            user_input: Запрос пользователя
            script_path: Путь к скрипту
            mode: Принудительный режим работы
            bat_path: Путь к bat файлу
            results_file: Файл результатов
            project_path: Путь к проекту
            model_path: Путь к модели (для PREDICT)
            use_llm_parsing: Использовать ли LLM для парсинга (False = только базовые правила)
        """

        if use_llm_parsing:
            try:
                params = extract_parameters_with_fallback(user_input)
            except Exception as e:
                logging.warning(f"LLM парсинг неудачен: {e}. Используем базовые правила")
                params = StableConfigFactory._extract_params_basic(user_input)
        else:
            params = StableConfigFactory._extract_params_basic(user_input)

        if mode:
            params["mode"] = mode

        config = create_minimal_config(
            params=params,
            script_path=script_path,
            project_path=project_path,
            bat_path=bat_path,
            results_file=results_file,
            model_path=model_path
        )

        if use_llm_parsing and os.path.exists(script_path):
            try:
                from .json_creator import ConfigTools
                variables = ConfigTools.parse_script(script_path)
                config["variables"] = variables
                logging.info("Скрипт успешно распарсен через LLM")
            except Exception as e:
                logging.warning(f"LLM парсинг скрипта неудачен: {e}. Используем простой парсинг")
                variables = parse_script_simple(script_path)
                config["variables"] = variables
        else:
            variables = parse_script_simple(script_path)
            config["variables"] = variables

        config = StableConfigFactory._correct_bounds_simple(user_input, config)

        config = StableConfigFactory._finalize_config_by_mode(config, params)

        logging.info(f"Конфигурация создана успешно. Режим: {params['mode']}, Тип: {params['rom_type']}")
        return config

    @staticmethod
    def _extract_params_basic(user_input: str) -> Dict:
        text = user_input.lower()

        if "предсказ" in text and "обуч" in text:
            mode = "TP"
        elif "предсказ" in text:
            mode = "PREDICT"
        else:
            mode = "TRAIN"

        if "nn" in text or "нейрон" in text:
            rom_type = "NN"
        elif "gpr" in text or "гаусс" in text:
            rom_type = "GPR"
        else:
            rom_type = "SVD"

        numbers = re.findall(r'\b(\d+)\b', user_input)
        n_snapshots = int(numbers[0]) if numbers else 100

        # DOE тип
        doe_type = "MEPE" if "адаптив" in text or "mepe" in text else "LHS"

        return {
            "mode": mode,
            "rom_type": rom_type,
            "n_snapshots": min(n_snapshots, 1000),  # Ограничиваем разумными пределами
            "doe_type": doe_type,
            "number_of_cores": 4,
            "needed_params": None
        }

    @staticmethod
    def _correct_bounds_simple(user_input: str, config: Dict) -> Dict:
        """Простая коррекция границ без LLM"""
        bounds_patterns = [
            r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]',
            r'от\s+(\d+\.?\d*)\s+до\s+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)'
        ]

        found_bounds = None
        for pattern in bounds_patterns:
            match = re.search(pattern, user_input)
            if match:
                found_bounds = [float(match.group(1)), float(match.group(2))]
                break

        if found_bounds:
            for var in config.get("variables", {}).get("input_variables", []):
                var["bounds"] = found_bounds
            logging.info(f"Установлены границы: {found_bounds}")

        return config

    @staticmethod
    def _finalize_config_by_mode(config: Dict, params: Dict) -> Dict:
        """Финальная настройка конфигурации в зависимости от режима"""
        mode = params["mode"]

        if mode == "PREDICT":
            # Убираем ненужные параметры для режима предсказания
            config["experiment"].pop("n_snapshots", None)
            config["experiment"].pop("doe_type", None)
            config["project"].pop("script_paths", None)
            config["project"].pop("bat_path", None)
            config["variables"].pop("input_variables", None)

            output_vars = []
            for var in config["variables"]["output_variables"]:
                output_vars.append({"name": var["name"]})
            config["variables"]["output_variables"] = output_vars

            config["rom_settings"] = {"rom_type": params["rom_type"]}

        return config

    @staticmethod
    def save_config(config: Dict, output_path: str) -> None:
        """Сохранение конфигурации в файл"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logging.info(f"Конфигурация сохранена: {output_path}")


def create_config_for_agent(query: str, **kwargs) -> str:
    """
    Функция-обертка для использования в агенте ROM
    """
    try:
        config = StableConfigFactory.create_config(
            user_input=query,
            script_path=kwargs.get("script_path", "/default/script.txt"),
            project_path=kwargs.get("project_path", "/default/project"),
            bat_path=kwargs.get("bat_path", "/default/batch.bat"),
            results_file=kwargs.get("results_file", "result.txt"),
            model_path=kwargs.get("model_path"),
            use_llm_parsing=True
        )

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            config_path = f.name

        return f"Конфигурация создана и сохранена: {config_path}"

    except Exception as e:
        logging.error(f"Ошибка создания конфигурации: {e}")

        try:
            config = StableConfigFactory.create_config(
                user_input=query,
                script_path=kwargs.get("script_path", "/default/script.txt"),
                project_path=kwargs.get("project_path", "/default/project"),
                use_llm_parsing=False
            )

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                config_path = f.name

            return f"Конфигурация создана (резервный режим): {config_path}"

        except Exception as e2:
            return f"Ошибка создания конфигурации: {str(e2)}"