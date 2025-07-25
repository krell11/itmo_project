from .db_config import ChromaConfig
from typing import Optional, List
from langchain_chroma import Chroma
from uuid import uuid4
from langchain.schema import Document
import chromadb
from chromadb.config import Settings


class VectorDataBaseManager:
    def __init__(self, config: ChromaConfig):
        self.config = config
        self.vectorstore: Optional[Chroma] = None
        self._client = chromadb.PersistentClient(path="E:/llm-rom/chroma_data",
                                                 settings=Settings(anonymized_telemetry=False))
        print(self._client.heartbeat())

    @property
    def is_initialized(self) -> bool:
        return self.vectorstore is not None

    def create_from_documents(self, documents: List[Document], collection_name: str):
        self.vectorstore = Chroma(client=self._client,
                                  collection_name=collection_name,
                                  client_settings=self.config.client_settings,
                                  embedding_function=self.config.embedding_model)
        self.vectorstore.add_documents(documents=documents, ids=[str(uuid4()) for _ in range(len(documents))])

    def add_to_collection(self, documents: List[Document], collection_name: str):
        existing_store = Chroma(
            collection_name=collection_name,
            persist_directory=self.config.persist_dir,
            embedding_function=self.config.embedding_model
        )
        existing_store.add_documents(documents)

    def collection_exists(self, collection_name: str) -> bool:
        try:
            self._client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_retriever(self, collection_name: str, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 4}

        store = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self.config.embedding_model
        )
        return store.as_retriever(search_kwargs=search_kwargs)

    def list_collections(self) -> List[str]:
        return list(self._client.list_collections())
