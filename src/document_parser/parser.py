from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
import re


def clean_text(text, max_newlines: int = 1):
    pattern = r'\n{' + str(max_newlines + 1) + ',}'
    replacement = '\n' * max_newlines
    cleaned_text = re.sub(pattern, replacement, text)
    return cleaned_text


class Parser:
    def __init__(self, chunk_size: int = 480, chunk_overlap: int = 20, length_function=len,
                 separators=None):

        self.loader = None
        self.splitter = None
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators

    def init_text_splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap,
                                                       length_function=len,
                                                       separators=self.separators)

    def init_pipeline(self, path_to_doc: str):
        self.init_text_splitter()
        if path_to_doc.endswith(".doc") or path_to_doc.endswith(".docx"):
            loaded_doc = Docx2txtLoader(path_to_doc)
        elif path_to_doc.endswith(".pdf"):
            loaded_doc = PyPDFLoader(path_to_doc)
        else:
            raise NotImplemented
        data = loaded_doc.load()
        data[0].page_content = clean_text(data[0].page_content)
        chunks = self.splitter.split_documents(data)
        return chunks
