import os
import re
from agent.llm_manager import get_llm
from tools.agent_tools import list_files, read_file, write_file
from tools.chat_with_uploaded_docs import ask_question_from_uploaded_doc
from code_assistant import chain as code_chain

def display_welcome():
    print("""
🖱️ Hello! I'm Cursor-DevAgent, your coding assistant within the IDE. I'm ready to help with:

🔹 File operations: List, read, write files
🔹 Code understanding: Explain code sections, search codebase
🔹 Development tasks: Generate code snippets, help with Python/CLI tools
🔹 IDE integration: Work within your VS Code context

I'll use only the available tools and give you minimal, functional answers that fit your coding style.
I won't suggest GUIs or external extensions unless you ask.

What would you like me to help with today?
➡️ Explore your current file structure
➡️ Explain code in open files
➡️ Help with specific coding tasks
➡️ Search through your codebase
➡️ Assist with file operations

Just type your command below ⬇️
""")

def extract_filename_from_query(query):
    """Extract filename from natural language query"""
    # Common patterns for filenames
    patterns = [
        r'file\s+([a-zA-Z0-9_\-\.]+)',
        r'read\s+([a-zA-Z0-9_\-\.]+)',
        r'write\s+([a-zA-Z0-9_\-\.]+)',
        r'create\s+([a-zA-Z0-9_\-\.]+)',
        r'open\s+([a-zA-Z0-9_\-\.]+)',
        r'([a-zA-Z0-9_\-\.]+\.py)',
        r'([a-zA-Z0-9_\-\.]+\.txt)',
        r'([a-zA-Z0-9_\-\.]+\.md)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def create_file_with_content(filename, content):
    """Create a new file with specified content"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ Created file '{filename}' successfully!"
    except Exception as e:
        return f"❌ Error creating file: {e}"

def process_query(query):
    """Process user queries and execute appropriate actions"""
    query_lower = query.lower().strip()
    
    # Greeting
    if any(word in query_lower for word in ['hi', 'hello', 'hey']) and len(query_lower.split()) <= 3:
        return "Hello! How can I help you with your code today?"
    
    # File listing
    if any(word in query_lower for word in ['list', 'show', 'files', 'directory', 'what files']):
        try:
            files = list_files(".")
            return f"📁 Current directory contents:\n{files}"
        except Exception as e:
            return f"❌ Error listing files: {e}"
    
    # File reading
    elif any(word in query_lower for word in ['read', 'open', 'show content']):
        filename = extract_filename_from_query(query)
        if filename:
            try:
                content = read_file(filename)
                return f"📖 Content of '{filename}':\n{content}"
            except Exception as e:
                return f"❌ Error reading file: {e}"
        else:
            return "Please specify which file to read. Example: 'read main.py'"
    
    # File creation with content
    elif any(word in query_lower for word in ['create', 'make', 'new file']):
        # Java file creation
        if 'java' in query_lower:
            filename = extract_filename_from_query(query) or "HelloWorld.java"
            content = '''public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}'''
            return create_file_with_content(filename, content)
        
        # Python file creation
        elif 'python' in query_lower or 'py' in query_lower:
            filename = extract_filename_from_query(query) or "hello.py"
            content = '# Python script\nprint("Hello from Python!")\n'
            return create_file_with_content(filename, content)
        
        # Hello world in any language
        elif 'hello world' in query_lower or 'print' in query_lower:
            filename = extract_filename_from_query(query) or "hello.py"
            content = 'print("Hello, World!")\n'
            return create_file_with_content(filename, content)
        
        # Default file creation
        else:
            filename = extract_filename_from_query(query) or "new_file.txt"
            content = "# New file created by Cursor-DevAgent\n"
            return create_file_with_content(filename, content)
    
    # Code explanation
    elif any(word in query_lower for word in ['explain', 'what does', 'how does', 'code', 'function', 'class']):
        try:
            response = code_chain.invoke(query)
            return f"🤖 Code explanation:\n{response}"
        except Exception as e:
            return f"❌ Error explaining code: {e}"
    
    # Document Q&A
    elif any(word in query_lower for word in ['document', 'doc', 'ask', 'question about']):
        try:
            response = ask_question_from_uploaded_doc(query)
            return f"📚 Document answer:\n{response}"
        except Exception as e:
            return f"❌ Error with document query: {e}"
    
    # Help
    elif 'help' in query_lower:
        return """
🔧 Available commands:
• "list files" - Show current directory contents
• "read filename" - Read a specific file (e.g., "read main.py")
• "create hello world file" - Create a new file with hello world
• "create java file" - Create a Java HelloWorld program
• "explain this code" - Ask about codebase
• "what does this function do" - Explain specific code
• "help" - Show this help
• "exit" - Quit

💡 Examples:
• "create a new file with print hello world"
• "create java file with hello world"
• "read main.py"
• "explain the main function"
• "list all files in current directory"
"""
    
    # Default response with suggestions
    else:
        return f"🤔 I didn't understand: '{query}'\n\nTry these commands:\n• 'list files' - See what files are here\n• 'read main.py' - Read a file\n• 'create java file with hello world' - Make a Java file\n• 'help' - Show all commands"

def main():
    display_welcome()
    
    while True:
        try:
            query = input("\n➡️  You: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            elif not query:
                continue
            
            # Process the query
            response = process_query(query)
            print(f"🤖 {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 