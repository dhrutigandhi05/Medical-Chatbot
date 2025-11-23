from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, chunk_docs, create_embedding_model

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_files(data='data/')
minimal_docs = filter_to_minimal_docs(extracted_data)
chunks = chunk_docs(minimal_docs)
embedding = create_embedding_model()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key) # initialize pinecone client

index_name = "medical-chatbot"

# create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, # higher dimension = more accurate embeddings and more info
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# create vector store from documents
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    index_name=index_name
)