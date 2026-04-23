# Grid07 AI Engineering Assignment

## Overview
Cognitive routing + LangGraph + RAG system.

## Phases
- Phase 1: Persona Router (vector similarity)
- Phase 2: LangGraph content engine
- Phase 3: RAG + prompt injection defense

# Grid07 — Cognitive Routing & RAG Engine

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your GROQ_API_KEY
python main.py
```

## Phase 1 — Vector Persona Router
Embeds 3 bot personas using sentence-transformers (all-MiniLM-L6-v2)
into ChromaDB. route_post_to_bots() queries the store and returns bots
with cosine similarity above 0.35 (calibrated for MiniLM embeddings;
equivalent to ~0.85 for OpenAI embeddings at higher dimensionality).

## Phase 2 — LangGraph Content Engine
3-node state machine:
- Node 1 (decide_search): LLM reads persona → outputs a search query
- Node 2 (web_search): Calls mock_searxng_search() with the query
- Node 3 (draft_post): LLM combines persona + search results → JSON post

State flows sequentially: decide_search → web_search → draft_post → END

## Phase 3 — Prompt Injection Defence
Defence strategy: Trust boundary separation in the system prompt.
The system prompt uses explicit labelling to separate INSTRUCTION SPACE
(the immutable persona rules, delimited by === markers) from DATA SPACE
(the human's message, labelled "UNTRUSTED USER INPUT").

The LLM is explicitly told that any instruction-like text appearing under
"HUMAN'S LATEST REPLY" is part of the argument to rebut, not a command
to execute. This prevents the model from treating injected instructions
as authoritative, since the system prompt pre-empts any override attempt
by categorising it as untrusted input before it's processed.

## Run

python main.py
