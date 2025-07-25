from src.database import VectorDataBaseManager, ChromaConfig
from langchain_huggingface import HuggingFaceEmbeddings
from src.document_parser import Parser


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(
                                            model_name="intfloat/multilingual-e5-small",
                                            model_kwargs={"device": "cuda"},
                                            encode_kwargs={"normalize_embeddings": True})
    config = ChromaConfig(embedding_model=embedding_model)
    data_base = VectorDataBaseManager(config=config)

    doc_parser = Parser()

    path = "C:\\work\\opt\\llm-rom\\docs\\CML_Bench_Руководство_оператора_v17.docx"
    chunks = doc_parser.init_pipeline(path)
    data_base.create_from_documents(documents=chunks, collection_name="rukovod")

