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

    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True})

    db_config = ChromaConfig(embedding_model=embedding_model)
    db_manager = VectorDataBaseManager(db_config)

    pipeline = RAGPipeline(llm_client, db_manager)


    async def main():
        while True:
            try:
                query = input("\nВведите вопрос (или 'exit' для выхода): ")
                if query.lower() in ('exit', 'quit'):
                    break

                result = await pipeline.run(query)
                print("\nОтвет:", result["answer"])
                print("\nКонтекст:", result['context'])
            except KeyboardInterrupt:
                print("\nЗавершение работы...")
                break


    asyncio.run(main())
