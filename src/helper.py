from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# extract text from pdfs
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    docs = loader.load()
    return docs

# filter documents to only keep content and source metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    
    # iterate through each document and get the content and source metadata
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs

# chunking the documents
def chunk_docs(minimal_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 500 characters = 1 chunk
        chunk_overlap=20, # 20 characters overlap between chunks
    )

    chunks = splitter.split_documents(minimal_docs)
    return chunks

# create embedding model
def create_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )

    return embeddings