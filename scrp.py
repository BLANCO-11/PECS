from ddgs import DDGS
from typing import List, Dict


def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict]:
    """
    Perform a DuckDuckGo search and return structured results.
    """
    results = []

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            results.append({
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body"),
            })

    return results


if __name__ == "__main__":
    query = input("Enter search query: ")
    search_results = duckduckgo_search(query, max_results=5)

    if not search_results:
        print("No results found.")
    else:
        for i, result in enumerate(search_results, start=1):
            print(f"\nResult {i}")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")