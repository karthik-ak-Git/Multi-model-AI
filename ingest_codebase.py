# ingest_codebase.py

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PythonLoader


def load_and_process_files():
    """
    Finds, loads, and processes all relevant files in the current directory,
    respecting a defined ignore list.
    """
    # 1. Define patterns to ignore
    # We will ignore the virtual environment, pycache, git history, and our own DB
    ignore_patterns = [
        "venv/",
        "__pycache__/",
        ".git/",
        "chroma_db_codebase/",
        "node_modules/",
        "__init__.py",  # Often not useful for context
        ".env",
        "docs/"
    ]

    all_documents = []
    print("[INFO] Starting file discovery...")

    # 2. Walk through the directory tree
    for root, dirs, files in os.walk("."):
        # Check if the current directory should be ignored
        # We modify dirs in-place to prevent os.walk from descending into them
        dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(
            p) for p in [os.path.normpath(pat) for pat in ignore_patterns])]

        for file in files:
            file_path = os.path.join(root, file)

            # Skip if the file path matches any ignore pattern
            if any(os.path.normpath(file_path).startswith(p) for p in [os.path.normpath(pat) for pat in ignore_patterns]):
                continue

            # 3. Load the file based on its extension
            try:
                if file.endswith(".py"):
                    loader = PythonLoader(file_path)
                    all_documents.extend(loader.load())
                elif file.endswith(".md") or file.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    all_documents.extend(loader.load())
            except Exception as e:
                print(f"[WARN] Error loading file {file_path}: {e}")

    print(f"[INFO] Loaded {len(all_documents)} relevant documents.")
    return all_documents

# --- Main Ingestion Logic ---


# 1. Load all relevant documents using our new function
documents = load_and_process_files()

# 2. Split documents into smaller chunks
print("[INFO] Splitting documents into chunks...")
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", chunk_size=1000, chunk_overlap=100
)
chunks = python_splitter.split_documents(documents)
print(f"[INFO] Created {len(chunks)} chunks.")

# 3. Embed and Store
print("[INFO] Generating embeddings and storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = "./chroma_db_codebase"

db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
db.persist()

print(f"[✅] Successfully ingested codebase into {persist_dir}")
