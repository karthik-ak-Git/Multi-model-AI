# tools/chat_with_uploaded_docs.py
from langchain.chains import RetrievalQA
from agent.llm_manager import get_llm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def ask_question_from_uploaded_doc(query, user_id="user_001"):
    persist_dir = f"./chroma_db/{user_id}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(persist_directory=persist_dir,
                         embedding_function=embeddings)

    retriever = vectorstore.as_retriever()
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": query})

    return result["result"]
