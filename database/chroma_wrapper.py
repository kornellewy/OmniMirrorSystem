from pathlib import Path
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


class ChromaWrapper:
    OPENAI_MODELS_NAMES = ["text-embedding-3-small"]

    def __init__(self, config: dict) -> None:
        self.config = config
        if self.config["text_embedding_model"] in self.OPENAI_MODELS_NAMES:
            embeddings = OpenAIEmbeddings(model=self.config["text_embedding_model"])
            self.vector_db = Chroma(
                embedding_function=embeddings,
                persist_directory=Path(__file__).parent / "db",
            )
        else:
            pass

    def add_documents(self, scan_results: dict) -> bool:
        for file_path, file_metadata in tqdm(scan_results.items()):
            self.add_file_to_database(file_path, file_metadata)
        self.vector_db.persist()

    def add_file_to_database(self, file_path, metadata):
        # Load the document from the text file
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        # Split the document into smaller chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
        )
        docs = text_splitter.split_documents(documents)
        if len(docs) > 0:
            for doc in docs:
                doc.metadata = metadata
            self.vector_db.add_documents(docs)
            print(f"Uploaded {file_path} with metadata to the vector store.")
