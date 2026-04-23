# phase2_langgraph.py
# 3-node autonomous content engine: bots research then post.

import json, os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# --- STATE DEFINITION ---
# TypedDict enforces the keys in our shared state dict.
class PostState(TypedDict):
    persona: str          # Bot personality (input)
    bot_id: str           # Bot identifier (input)
    search_query: str     # Set by Node 1
    search_results: str   # Set by Node 2
    post: dict            # Final JSON — set by Node 3

# --- MOCK SEARCH TOOL ---
# Assignment requirement: hardcoded headlines based on keywords.
@tool
def mock_searxng_search(query: str) -> str:
    """Search for recent news headlines based on a query."""
    q = query.lower()
    if "crypto" in q or "bitcoin" in q:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals. Institutional investors pour $2B into crypto funds."
    elif "ai" in q or "openai" in q or "model" in q:
        return "OpenAI announces GPT-5 with reasoning surpassing PhD-level benchmarks. Anthropic Claude 4 follows days later."
    elif "market" in q or "stocks" in q or "interest" in q:
        return "Fed holds rates at 5.25%. S&P 500 up 2.3% on strong jobs data. Nasdaq leads tech rally."
    elif "privacy" in q or "surveillance" in q:
        return "EU passes AI Surveillance Act banning facial recognition in public spaces."
    else:
        return "AI investment hits record $50B in Q1 2024. Major tech layoffs continue."

# --- LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# --- NODE 1: DECIDE SEARCH QUERY ---
# LLM reads persona → outputs what topic to search today.
def node_decide_search(state: PostState) -> PostState:
    print(f"[Node 1] Deciding topic for {state['bot_id']}...")
    prompt = f"""You are a social media bot with this persona:
{state['persona']}

Based on your persona, decide ONE topic to post about today.
Respond with ONLY a short search query (3-6 words), nothing else."""
    response = llm.invoke(prompt)
    state["search_query"] = response.content.strip().replace('"', '')
    print(f"  Search query: {state['search_query']}")
    return state

# --- NODE 2: WEB SEARCH ---
# Calls mock search with the query from Node 1.
def node_web_search(state: PostState) -> PostState:
    print(f"[Node 2] Searching: '{state['search_query']}'...")
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    state["search_results"] = results
    print(f"  Context: {results[:80]}...")
    return state

# --- NODE 3: DRAFT POST (JSON OUTPUT) ---
# Persona + search results → opinionated 280-char post as JSON.
def node_draft_post(state: PostState) -> PostState:
    print(f"[Node 3] Drafting post...")
    prompt = f"""You are a social media bot with this persona:
{state['persona']}

Recent news context: {state['search_results']}

Write a highly opinionated post (UNDER 280 characters) about: {state['search_query']}

Respond with ONLY a valid JSON object in this exact format — no markdown, no explanation:
{{"bot_id": "{state['bot_id']}", "topic": "...", "post_content": "..."}}"""
    response = llm.invoke(prompt)
    raw = response.content.strip().strip('`')
    if raw.startswith('json'): raw = raw[4:]
    state["post"] = json.loads(raw)
    print(f"  Post: {state['post']['post_content'][:60]}...")
    return state

# --- ASSEMBLE THE GRAPH ---
def build_content_graph():
    graph = StateGraph(PostState)
    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search", node_web_search)
    graph.add_node("draft_post", node_draft_post)
    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)
    return graph.compile()

if __name__ == "__main__":
    app = build_content_graph()
    result = app.invoke({
        "bot_id": "bot_a_tech_maximalist",
        "persona": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
        "search_query": "",
        "search_results": "",
        "post": {},
    })
    print("\\n[FINAL JSON OUTPUT]")
    print(json.dumps(result["post"], indent=2))