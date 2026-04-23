# phase1_router.py
# Stores bot persona embeddings in ChromaDB.
# Matches incoming posts to the right bots via cosine similarity.

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# --- Bot Personas ---
# The richer the description, the better the semantic matching.
BOT_PERSONAS = [
    {
        "id": "bot_a_tech_maximalist",
        "persona": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        )
    },
    {
        "id": "bot_b_doomer_skeptic",
        "persona": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        )
    },
    {
        "id": "bot_c_finance_bro",
        "persona": (
            "I strictly care about markets, interest rates, trading algorithms, "
            "and making money. I speak in finance jargon and view everything "
            "through the lens of ROI."
        )
    },
]

# --- Embedding Function ---
# sentence-transformers runs locally (free, no API key needed).
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-MiniLM-L3-v2"
    )

# --- Build ChromaDB in-memory store ---
# EphemeralClient = RAM only, no disk. Simulates pgvector per assignment.
def build_persona_store():
    client = chromadb.EphemeralClient()
    ef = get_embedding_fn()

    collection = client.create_collection(
        name="bot_personas",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}  # Cosine distance metric
    )

    collection.add(
        documents=[b["persona"] for b in BOT_PERSONAS],
        ids=[b["id"] for b in BOT_PERSONAS],
    )
    print(f"[Phase 1] Loaded {len(BOT_PERSONAS)} bot personas into vector store.")
    return collection

# --- Main Routing Function ---
# ChromaDB returns distance (0=identical, 2=opposite).
# Convert: similarity = 1 - (distance / 2) → range 0 to 1
def route_post_to_bots(post_content: str, collection, threshold: float = 0.35):
    results = collection.query(
        query_texts=[post_content],
        n_results=3
    )

    matched_bots = []
    for bot_id, distance in zip(results["ids"][0], results["distances"][0]):
        similarity = 1 - (distance / 2)
        if similarity >= threshold:
            matched_bots.append({"bot_id": bot_id, "similarity": round(similarity, 4)})
            print(f"  MATCHED: {bot_id} (similarity: {similarity:.4f})")
        else:
            print(f"  SKIPPED: {bot_id} (similarity: {similarity:.4f} < {threshold})")

    return matched_bots


if __name__ == "__main__":
    store = build_persona_store()
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits $100K as SEC approves spot ETF applications.",
        "Tech companies are destroying mental health with addictive algorithms.",
    ]
    for post in test_posts:
        print(f"\n[POST]: {post}")
        route_post_to_bots(post, store)