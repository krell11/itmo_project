from src.database.vec_storage import VectorDataBaseManager
from src.database.db_config import ChromaConfig
from src.llm.llm_base.llm_client import LlmClient
from src.llm.rag_pipeline import RAGPipeline

from langchain_huggingface import HuggingFaceEmbeddings

import asyncio


if __name__ == "__main__":
    llm_client = LlmClient(
        api_base="http://localhost:8000/v1",
        model_name="/models/Qwen2-5-7B-Q4_K_M/Qwen2.5-7B.Q4_K_M.gguf",
        temperature=0.1)
