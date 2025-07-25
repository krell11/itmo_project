from src.database import VectorDataBaseManager, ChromaConfig
from langchain_huggingface import HuggingFaceEmbeddings
from src.document_parser import Parser
import os


def get_collections_name(path: str) -> list:
    names = os.listdir(path)
    return names


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True})
    config = ChromaConfig(embedding_model=embedding_model)
    data_base = VectorDataBaseManager(config=config)
    base_path = "E:\llm-rom\docs"
    db_collections = get_collections_name(base_path)
    doc_parser = Parser()
    for collection_name in db_collections:
        collection_path = os.path.join(base_path, collection_name)
        docs_names = get_collections_name(collection_path)
        chunks = []
        for idx, doc_name in enumerate(docs_names):
            full_doc_path = os.path.join(collection_path, doc_name)
            chunks.extend(doc_parser.init_pipeline(full_doc_path))
        data_base.create_from_documents(documents=chunks, collection_name=collection_name)
