# test_rag_query.py

from tools.web_search import duckduckgo_search
from tools.refine_with_llm import summarize_search_results

query = "What is the difference between AI and Machine Learning?"

search_data = duckduckgo_search(query)
final_answer = summarize_search_results(search_data, query)

print("\n🧠 Final Answer:\n")
print(final_answer)
