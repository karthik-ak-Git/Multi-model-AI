# cursor_agent.py

import os
import re
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field
from agent.llm_manager import get_llm

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ==============================================================================
#  TOOL DEFINITIONS
# ==============================================================================

def _clean_path(path: str) -> str:
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
#  AGENT SETUP
# ==============================================================================


llm = get_llm(temperature=0)
tools = [list_files, read_file, write_file]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant that can write and read files."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==============================================================================
#  INTERACTIVE CHAT
# ==============================================================================

if __name__ == "__main__":
    print("🤖 AI Code Agent is ready.")
    print("Type 'exit' to quit.")

    while True:
        try:
            query = input("➡️  You: ")
            if query.lower() == 'exit':
                break

            print("\n🤖 Assistant:")
            result = agent_executor.invoke({"input": query})
            print(result.get('output'))
            print("\n")

        except (KeyboardInterrupt, EOFError):
            break
