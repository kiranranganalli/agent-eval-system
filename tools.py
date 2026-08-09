from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


import re

def _slugify(text: str, max_words: int = 5) -> str:
    """Turn a query into a clean filename fragment, e.g.
    'side effects of over usage refrigerator' -> 'side_effects_over_usage_refrigerator'"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)        # strip punctuation
    words = text.split()
    stopwords = {"of", "the", "a", "an", "is", "are", "what", "why", "how", "does", "do"}
    words = [w for w in words if w not in stopwords]
    return "_".join(words[:max_words]) if words else "query"


def save_to_txt(data: str, filename: str = None, topic: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if filename is None:
        if topic:
            slug = _slugify(topic)
            filename = f"research_{slug}_{timestamp}.txt"
        else:
            filename = f"research_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)


# Wikipedia wrapped so a flaky API returns a message instead of crashing the run
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)


def safe_wikipedia(query: str):
    try:
        return api_wrapper.run(query)
    except Exception as e:
        return f"Wikipedia lookup failed (API error): {e}"


wiki_tool = Tool(
    name="wikipedia",
    func=safe_wikipedia,
    description="Look up information on Wikipedia",
)