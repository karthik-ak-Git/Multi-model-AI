import os
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langsmith.run_helpers import traceable  # Keep this for LangSmith tracing

# Load environment variables
load_dotenv()

VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dev-agent")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west-2")


@traceable(name="Document Ingestion")
def ingest():
    print("[INFO] Loading documents...")
    loader = TextLoader("./docs/sample1.txt")
    documents = loader.load()

    print("[INFO] Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20
    )
    documents = text_splitter.split_documents(documents)

    print("[INFO] Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if VECTOR_STORE == "pinecone":
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_community.vectorstores import Pinecone as PineconeVectorStore

            pc = Pinecone(api_key=PINECONE_API_KEY)

            if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=512,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
                )

            print("[INFO] Storing documents in Pinecone...")
            vectorstore = PineconeVectorStore.from_documents(
                documents,
                embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace="default"
            )

            print(f"[INFO] Stored {len(documents)} docs in Pinecone.")
            return

        except Exception as e:
            print(f"[WARN] Pinecone failed: {e} — falling back to ChromaDB")

    try:
        from langchain_community.vectorstores import Chroma

        print("[INFO] Storing documents in ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory="./chroma_db"
        )
        print(f"[INFO] Stored {len(documents)} docs in ChromaDB.")

    except Exception as e:
        print(f"[ERROR] Failed to store documents in ChromaDB: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("[INFO] Starting ingestion...")
    ingest()
