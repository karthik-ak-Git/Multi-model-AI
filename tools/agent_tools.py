# cursor_agent.py

import os
import re
from langchain import hub
# New agent constructor
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from pydantic.v1 import BaseModel, Field
from agent.llm_manager import get_llm

# ==============================================================================
#  TOOL DEFINITIONS
# ==============================================================================


def _clean_path(path: str) -> str:
    """A helper function to clean up file paths from LLM output."""
    path = re.sub(r"\(.*\)", "", path).strip()
    path = path.strip("'\"` ")
    return path


@tool
def list_files(directory: str = ".") -> str:
    """Lists all files and directories in a given directory."""
    cleaned_directory = _clean_path(directory)
    try:
        return "\n".join(os.listdir(cleaned_directory))
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_file(file_path: str) -> str:
    """Reads the entire content of a specified file."""
    cleaned_path = _clean_path(file_path)
    try:
        with open(cleaned_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


class WriteFileInput(BaseModel):
    file_path: str = Field(description="The path of the file to write to.")
    content: str = Field(description="The full content to write to the file.")


@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str) -> str:
    """Writes or overwrites the content of a specified file."""
    cleaned_path = _clean_path(file_path)
    try:
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {cleaned_path}."
    except Exception as e:
        return f"Error writing to file: {e}"

# ==============================================================================
#  AGENT SETUP (UPGRADED)
# ==============================================================================


# 1. Initialize the LLM
llm = get_llm(temperature=0)

# 2. Define the list of all available tools
# We will add the retriever tool separately for some models
tools = [list_files, read_file, write_file]

# 3. Get the modern prompt template for tool calling
prompt = hub.pull("hwchase17/xml-agent-convo")

# 4. Create the agent using the new constructor
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. Create the AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ==============================================================================
#  INTERACTIVE CHAT
# ==============================================================================

if __name__ == "__main__":
    print("🤖 AI Code Agent is ready. It can read, write, and search files.")
    print("Type 'exit' to quit.")

    while True:
        try:
            query = input("➡️  You: ")
            if query.lower() == 'exit':
                break

            print("\n🤖 Assistant:")
            result = agent_executor.invoke(
                {"input": query, "chat_history": []})
            print(result.get('output'))
            print("\n")

        except (KeyboardInterrupt, EOFError):
            break
