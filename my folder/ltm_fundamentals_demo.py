"""
LangGraph Memory Store — Basics & Operations

Theme: Memory Store is a persistent, structured key-value store
       that operates OUTSIDE the message thread.

Core Insight:
- Checkpointer  = conversation memory (what was said)
- Memory Store  = user/entity memory (what is known)
- Analogy : checkpointer = chat history
            store        = user profile database

Key Operations:
- put()    → write data
- get()    → read by exact key
- search() → semantic search across stored data
"""

from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
from langchain_cohere import CohereEmbeddings

load_dotenv()

# ==================================================
# PART 1: BASIC STORE OPERATIONS
# Demonstrating put(), get(), search()
# No embeddings — pure key-value storage
# ==================================================

print("\n" + "=" * 20 + " PART 1: BASIC OPERATIONS " + "=" * 20)

store = InMemoryStore()

# --------------------------------------------------
# Namespaces — Logical Isolation
# Format: (category, entity_id)
# Different namespaces = completely isolated data
# Same key in different namespaces = different records
# --------------------------------------------------

# User 1 data — isolated to ('users', 'user1')
namespace_user1 = ("users", "user1")

store.put(namespace=namespace_user1, key="1", value={"data": "user is a coder."})
store.put(namespace=namespace_user1, key="2", value={"data": "user codes in python."})

# User 2 data — isolated to ('users', 'user2')
# Completely separate from user1 despite same keys
namespace_user2 = ("users", "user2")

store.put(namespace=namespace_user2, key="1", value={"data": "user is british."})
store.put(namespace=namespace_user2, key="2", value={"data": "user likes soccer."})

# --------------------------------------------------
# get() — Exact key lookup
# Retrieves one specific record by namespace + key
# --------------------------------------------------

print("\n--- get(): Exact Key Lookup ---")

# Fetch user2's key "1" — should return british nationality
result_user2 = store.get(namespace=namespace_user2, key="1").value
print(f"User2, key=1 → {result_user2}")

# Fetch user1's key "2" — should return python coding fact
result_user1 = store.get(namespace=namespace_user1, key="2").value
print(f"User1, key=2 → {result_user1}")

# --------------------------------------------------
# search() — Scan all records in a namespace
# Returns all items stored under the namespace
# No semantic understanding — pure key-value scan
# --------------------------------------------------

print("\n--- search(): Full Namespace Scan ---")

items = store.search(namespace_user1)
print(f"All records for user1 ({len(items)} items):")

for item in items:
    print(f"  key={item.key} → {item.value}")

# ==================================================
# PART 2: SEMANTIC SEARCH WITH EMBEDDINGS
# Upgrade store with vector index for similarity search
# Now search() understands MEANING not just keywords
# ==================================================

print("\n" + "=" * 20 + " PART 2: SEMANTIC SEARCH " + "=" * 20)

# --------------------------------------------------
# Embedding Setup
# Cohere embed-english-v3.0 → 1024 dimensional vectors
# Each stored string becomes a searchable vector
# --------------------------------------------------

embed = CohereEmbeddings(model="embed-english-v3.0")

# Reinitialise store with vector index
# dims=1024 must match embedding model output dimensions
store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1024
    }
)

# --------------------------------------------------
# Store User Profile Facts
# Each fact = one record = one searchable vector
# Real-world use: personalisation, context injection
# --------------------------------------------------

user_profile_facts = [
    "User prefers concise answers over long explanations",
    "User likes examples in Python",
    "User usually works late at night",
    "User prefers dark mode in applications",
    "User is learning machine learning",
    "User dislikes overly theoretical explanations",
    "User prefers step-by-step reasoning",
    "User is based in India",
    "User likes real-world analogies",
    "User prefers bullet points over paragraphs",
]

namespace_user3 = ("users", "user3")

# Keys are 1-indexed strings for clean retrieval
for key, fact in enumerate(user_profile_facts, 1):
    store.put(namespace_user3, str(key), {"data": fact})

print(f"\n Stored {len(user_profile_facts)} profile facts for user3")

# --------------------------------------------------
# get() — Exact lookup still works with embeddings
# --------------------------------------------------

print("\n--- get(): Exact Lookup (key='2') ---")

exact = store.get(namespace_user3, "2").value
print(f"key=2 → {exact}")

# --------------------------------------------------
# search() with query — Semantic Similarity Search
# Finds records CLOSEST IN MEANING to the query
# Does NOT require exact keyword match
# --------------------------------------------------

print("\n--- search(): Semantic Query — 'tell users nationality' ---")
print("Query has no exact keyword match — tests semantic understanding\n")

semantic_results = store.search(
    namespace_user3,
    query="tell users nationality",
    limit=3  # top 3 most semantically similar results
)

for i, item in enumerate(semantic_results, 1):
    print(f"  [{i}] key={item.key} → {item.value}")

# --------------------------------------------------
# Summary: Store Operations at a Glance
# --------------------------------------------------

print("\n" + "=" * 20 + " OPERATIONS SUMMARY " + "=" * 20)
print("""
┌─────────────┬───────────────────────────────────────────┐
│ Operation   │ Description                               │
├─────────────┼───────────────────────────────────────────┤
│ put()       │ Write key-value data to a namespace       │
│ get()       │ Read one record by exact namespace + key  │
│ search()    │ Scan namespace (+ semantic if embedded)   │
│ namespace   │ Logical isolation — (category, entity_id) │
└─────────────┴───────────────────────────────────────────┘

Checkpointer  → conversation history (what was SAID)
Memory Store  → entity profiles    (what is KNOWN)
""")