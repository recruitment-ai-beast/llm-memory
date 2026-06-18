# LLM Memory — Complete Guide

> Most developers think LLMs remember things.
> They don't. Here's what's actually happening — 
> and how to engineer memory that works.

---

## The Core Truth

```
LLM has NO memory.
Every response is stateless by default.
What feels like memory is engineering — not magic.
```

---

## Memory Types Covered

| Type | Mechanism | Persistence | Use Case |
|------|-----------|-------------|----------|
| Short-Term (RAM) | InMemorySaver | ❌ Lost on restart | Dev, testing |
| Persistent (DB) | SqliteSaver / PostgresSaver | ✅ Survives restarts | Production |
| Trimming | trim_messages | N/A — windowing only | Token cost control |
| Deletion | RemoveMessage | ✅ Permanent removal | Privacy, cleanup |
| Long-Term (LTM) | InMemoryStore + embeddings | ✅ Entity profiles | Personalisation |

---

## 1. Short-Term Memory (RAM)

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

**Behaviour:**
- Stores full conversation per `thread_id`
- Lives in RAM — wiped on runtime restart
- Perfect for development and testing

**Analogy:** Sticky note on your desk. Gone when you leave.

---

## 2. Persistent Memory (Database)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = builder.compile(checkpointer=checkpointer)
```

**Behaviour:**
- Writes conversation state to disk
- Survives runtime restarts
- Same `thread_id` = same conversation, always

**Analogy:** Notebook on your desk. Still there tomorrow.

---

## 3. Trimming — Windowing, Not Deletion

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

trimmed = trim_messages(
    state["messages"],
    token_counter=count_tokens_approximately,
    max_tokens=150,
    strategy="last"  # keep most recent
)

response = model.invoke(trimmed)  # LLM sees trimmed window only
```

**Behaviour:**
- Controls what gets SENT to LLM per turn
- Does NOT modify checkpointer state
- Full history remains intact in storage
- LLM may "forget" old context — storage never does

```
Checkpointer = Full library    (stores everything)
trim_messages = Your bag       (carry only what fits)
LLM = You                      (know only what you carry)
```

⚠️ **Trimming ≠ Deletion. Storage is always intact.**

---

## 4. Deletion — Permanent Removal

```python
from langchain_core.messages import RemoveMessage

def delete_old_messages(state: MessagesState):
    messages = state["messages"]
    if len(messages) > 10:
        to_delete = messages[:6]  # remove oldest 6
        return {"messages": [RemoveMessage(id=m.id) for m in to_delete]}
    return {}
```

**Behaviour:**
- Permanently removes messages from checkpointer
- No recovery. No undo.
- Reduces storage cost and enforces retention policies

```
Trimming  → closing a book chapter  (page still exists)
Deletion  → tearing the page out    (gone forever)
```

⚠️ **Deleted messages are unrecoverable.**

---

## 5. Long-Term Memory (LTM) — Memory Store

```python
from langgraph.store.memory import InMemoryStore
from langchain_cohere import CohereEmbeddings

embed = CohereEmbeddings(model="embed-english-v3.0")

store = InMemoryStore(index={"embed": embed, "dims": 1024})

namespace = ("users", "user1")

# Write
store.put(namespace, "1", {"data": "User is based in India."})

# Read exact
store.get(namespace, "1").value

# Semantic search
store.search(namespace, query="tell me users nationality", limit=3)
```

**Behaviour:**
- Operates OUTSIDE message threads
- Stores entity/user facts, not conversation history
- Semantic search finds relevant facts by meaning

```
Checkpointer  → what was SAID    (conversation history)
Memory Store  → what is KNOWN    (user/entity profiles)
```

---

## Memory Store Operations

```
┌─────────────┬───────────────────────────────────────────┐
│ Operation   │ Description                               │
├─────────────┼───────────────────────────────────────────┤
│ put()       │ Write key-value data to a namespace       │
│ get()       │ Read one record by exact namespace + key  │
│ search()    │ Scan namespace (+ semantic if embedded)   │
│ namespace   │ Logical isolation — (category, entity_id) │
└─────────────┴───────────────────────────────────────────┘
```

---

## The Full Picture

```
┌─────────────────────────────────────────────────────┐
│                   LangGraph Memory                  │
├──────────────────────┬──────────────────────────────┤
│   THREAD MEMORY      │      ENTITY MEMORY           │
│   (Checkpointer)     │      (Memory Store)          │
├──────────────────────┼──────────────────────────────┤
│ What was SAID        │ What is KNOWN                │
│ Conversation history │ User profiles, facts         │
│ Per thread_id        │ Per namespace                │
├──────────────────────┴──────────────────────────────┤
│              CONTEXT MANAGEMENT                      │
├──────────────────────┬──────────────────────────────┤
│ Trimming             │ Deletion                     │
│ Window control only  │ Permanent removal            │
│ Storage intact       │ Unrecoverable                │
└──────────────────────┴──────────────────────────────┘
```

---

## Stack

- **LangGraph** — graph execution + state management
- **LangChain** — LLM interface + message utilities
- **Groq** — LLM inference (`llama-3.1-8b-instant`)
- **SQLite** — persistent conversation storage
- **Cohere** — embeddings for semantic search
- **Rich** — terminal output formatting

---

## Run

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
```

Run any demo:
```bash
python my-folder/stm_demo.py
python my-folder/persistence_demo.py
python my-folder/trimming_demo.py
python my-folder/deletion_demo.py
python my-folder/ltm_fundamentals_demo.py
python my-folder/summarization_demo.py
```

---

## Part of

**BEAST — Vertical AI Engineer**
Building production AI systems for recruitment automation.
[LinkedIn](https://www.linkedin.com/in/beast-builds-ai) •
[Twitter](https://x.com/beastbuildsai) •  
[Live Product](https://ai-for-resume-screening.streamlit.app/)