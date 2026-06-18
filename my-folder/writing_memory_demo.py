"""
LangGraph Intelligent Memory Writer

Theme: Chatbot that writes its own memory — 
       deduplicates before storing.
       Only NEW, atomic facts get persisted.

Core Insight:
- LLM decides what's worth remembering
- Duplicate detection prevents memory bloat
- Each memory = one atomic fact (not full sentences)
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List
import uuid

load_dotenv()

# --------------------------------------------------
# Models
# --------------------------------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=200
)

# --------------------------------------------------
# Memory Schema
# LLM outputs structured decisions — not free text
# --------------------------------------------------

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user fact as a short sentence")
    is_new: bool = Field(description="True if NEW. False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether any new memories should be stored")
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="List of atomic facts extracted from user message"
    )

# Structured output — forces LLM to return MemoryDecision
extractor_model = model.with_structured_output(MemoryDecision)

# --------------------------------------------------
# Memory Prompt
# Gives LLM existing memory + asks for new facts
# Deduplication happens here — LLM marks is_new=False
# if fact already exists in known memory
# --------------------------------------------------

MEMORY_PROMPT = """\
You are a memory manager for a personal assistant.

Already known facts about the user:
{user_facts}

Your job:
1. Extract atomic facts from the user's message
2. Mark each as is_new=True only if NOT already known
3. Set should_write=False if nothing new to store

Be strict — no duplicates, no generic facts.\
"""

# --------------------------------------------------
# Store
# --------------------------------------------------

store = InMemoryStore()

# --------------------------------------------------
# Graph Node — Memory Writer
# Reads existing memory → extracts new facts → 
# deduplicates → writes only what's new
# --------------------------------------------------

def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Extract and store atomic user facts from each message.
    
    Flow:
    1. Load existing user memory from store
    2. Send existing memory + new message to extractor LLM
    3. LLM decides what's new vs duplicate
    4. Only new facts get written to store
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("users", user_id, "details")

    # Load existing memory for deduplication context
    existing_items = store.search(namespace)
    user_facts = (
        "\n".join(f"- {item.value.get('data', '')}" for item in existing_items)
        if existing_items else "No existing memory."
    )

    # Latest user message
    last_message = state["messages"][-1].content

    # LLM decides what to remember
    decision: MemoryDecision = extractor_model.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(user_facts=user_facts)),
        HumanMessage(content=last_message)
    ])

    # Write only new, non-duplicate facts
    if decision.should_write:
        new_count = 0
        for memory in decision.memories:
            if memory.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": memory.text})
                new_count += 1
        content = f"Noted ({new_count} new fact(s) stored)"
    else:
        content = "Nothing new to remember."

    return {"messages": [{"role": "assistant", "content": content}]}

# --------------------------------------------------
# Build Graph
# --------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

graph = builder.compile(store=store)

# --------------------------------------------------
# Run — includes deliberate duplicate to test dedup
# --------------------------------------------------

print("=" * 20 + " MEMORY WRITER DEMO " + "=" * 20)

conversation = [
    "HI! My name is Beast.",
    "HI! I am from INDIA.",
    "I am a Coder.",
    "I am learning LangGraph.",
    "Tell me about short term memory in LLMs.",
    "I am a Coder.",  # deliberate duplicate — should NOT be stored
]

config = {"configurable": {"user_id": "u3"}}

for user_input in conversation:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    )
    print(f"Human:     {user_input}")
    print(f"Assistant: {result['messages'][-1].content}\n")

# --------------------------------------------------
# Final Memory State — proof of deduplication
# --------------------------------------------------

print("=" * 20 + " STORED MEMORY " + "=" * 20)

items = store.search(("users", "u3", "details"))
print(f"Total facts stored: {len(items)}\n")

for i, item in enumerate(items, 1):
    print(f"[{i}] {item.value['data']}")

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))