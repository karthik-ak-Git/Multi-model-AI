# tools/refine_with_llm.py

from agent.llm_manager import get_llm


def summarize_search_results(search_results: str, query: str) -> str:
    llm = get_llm(temperature=0.7)

    prompt = f"""
You are a helpful AI assistant.

Here are some web search results about: "{query}".

Based on this, provide a concise and clear explanation in simple terms:

-----------------
{search_results}
-----------------

Your answer:
"""

    print("[🧠] Summarizing via Mistral...")
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else response
