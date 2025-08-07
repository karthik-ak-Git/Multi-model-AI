# code_assistant.py

import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import your flexible LLM loader
from agent.llm_manager import get_llm

# --- SETUP ---
# 1. Initialize the LLM using your manager
# Make sure your .env file is configured for the AI_PROVIDER you want to use.
llm = get_llm(temperature=0.1)

# 2. Load the vector store you created in the ingestion step
persist_dir = "./chroma_db_codebase"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_dir,
                     embedding_function=embeddings)

# 3. Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})

# --- PROMPT & CHAIN ---
# 4. Create a prompt template that instructs the LLM how to behave
template = """
You are an expert AI programming assistant.
Answer the user's question based *only* on the following context of source code files.
If you don't know the answer, just say that you don't know. Do not make up an answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate.from_template(template)

# 5. Create the RAG chain using LangChain Expression Language (LCEL)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- INTERACTIVE CHAT ---
# 6. Create a simple command-line interface
if __name__ == "__main__":
    print("🤖 AI Code Assistant is ready. Ask me anything about your codebase!")
    print("Type 'exit' to quit.")

    while True:
        try:
            query = input("➡️  You: ")
            if query.lower() == 'exit':
                break

            print("\n🤖 Assistant:")
            # Stream the response for a better user experience
            for chunk in chain.stream(query):
                print(chunk, end="", flush=True)
            print("\n")

        except (KeyboardInterrupt, EOFError):
            break
