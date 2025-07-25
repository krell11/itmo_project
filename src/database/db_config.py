from typing import Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaConfig:
    def __init__(self, persist_dir: str = '../chroma_data', collection_name: str = "default_collection",
                 embedding_model: Optional[HuggingFaceEmbeddings] = None,
                 client_settings: Optional[Dict[str, Any]] = None):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client_settings = client_settings
        