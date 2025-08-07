# tools/web_search.py

from duckduckgo_search import DDGS


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    print(f"[🔎] Searching DuckDuckGo for: {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            if r.get("body"):
                results.append(
                    f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}")
            else:
                results.append(f"Title: {r['title']}\nURL: {r['href']}")

    if not results:
        return "No relevant results found."

    return "\n\n".join(results)
