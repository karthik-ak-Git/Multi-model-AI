from llm_manager import get_llm

llm = get_llm()
response = llm.invoke("Hello, how are you?")
print(response)
