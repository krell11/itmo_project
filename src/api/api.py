from fastapi import FastAPI, HTTPException, Body, Depends, Header
import uuid
import logging

from src.rom_llm.agents.romagent import ROMAssistantAgent
from src.api.classes import QueryRequest, DirectConfigRequest, QueryResponse

app = FastAPI()

agent = None


logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_event():
    global agent

    from dotenv import load_dotenv
    import os
    load_dotenv("E:\\llm-rom\\.env")

    llm_api_base = os.getenv('API')
    llm_model_name = os.getenv('MODEL')

    chroma_db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    # postgres_connection_string = {"host": "localhost",
    #                               "port": 5432, "dbname": "rom_assistant", 'user': 'postgres', "password": 'postgres'}
    postgres_connection_string = "postgresql://postgres:postgres@localhost:5432/rom_assistant"
    agent = ROMAssistantAgent(
        llm_api_base=llm_api_base,
        llm_model_name=llm_model_name,
        chroma_db_path=chroma_db_path,
        postgres_connection_string=postgres_connection_string
    )


@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Обработка запроса пользователя
    """
    try:
        logger.info(f"Получен запрос от пользователя {request.user_id}: {request.query}")

        if request.config_path:
            logger.info(f"Предоставлен config_path: {request.config_path}")
        if request.config:
            logger.info(f"Предоставлен config в теле запроса")

        result = agent.process_query(
            user_id=request.user_id,
            chat_id=request.chat_id,
            query=request.query,
            config=request.config,
            config_path=request.config_path
        )

        logger.info(f"Запрос успешно обработан для пользователя {request.user_id}")

        return QueryResponse(
            response=result,
            status="success"
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return QueryResponse(
            response=f"Ошибка при обработке запроса: {str(e)}",
            status="error"
        )


@app.post("/api/run-config")
async def run_with_config(request: DirectConfigRequest):
    """
    Запуск ROM-модуля напрямую с конфигом, без использования агента
    """
    global agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    result = agent.run_with_config(config_json=request.config)

    return {
        "response": result
    }


@app.get("/api/health")
async def health_check():
    """
    Проверка здоровья API
    """
    return {"status": "ok"}


