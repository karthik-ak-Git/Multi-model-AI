import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PythonLoader, NotebookLoader, DirectoryLoader


def load_documents_from_directory(directory_path: str):
    loaders = [
        (".py", PythonLoader),
        (".txt", TextLoader),
        (".md", TextLoader),
        (".ipynb", NotebookLoader),
    ]

    documents = []
    for ext, LoaderClass in loaders:
        loader = DirectoryLoader(
            path=directory_path, glob=f"**/*{ext}", loader_cls=LoaderClass)
        docs = loader.load()
        documents.extend(docs)

    return documents
