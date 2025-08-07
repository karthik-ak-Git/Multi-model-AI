# tools/upload_and_ingest.py
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def ingest_file(file_path, user_id="user_001"):
    # 1. Load file
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # 3. Embed + Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = f"./chroma_db/{user_id}"  # Isolate user uploads

    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"[✅] Ingested {len(chunks)} chunks into {persist_dir}")
    return persist_dir
