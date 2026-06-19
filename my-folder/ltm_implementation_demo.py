"""
LangGraph Memory Writer + Reader Combined

Theme: Chatbot extracts and stores non-duplicate memory,
       then uses that same memory to respond intelligently.

Flow: remember (write) → chat (read + respond)
Two nodes, one namespace, one continuous memory loop.
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
# Schemas
# --------------------------------------------------

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if NEW. False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic facts to store")

# --------------------------------------------------
# Models
# --------------------------------------------------

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=150)
extractor_model = model.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """\
You are a memory manager. Already known facts:
{user_details_content}

Extract atomic facts from the user's message.
Mark is_new=True only if NOT already known.
Set should_write=False if nothing new.\
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant with knowledge about the user.

Known facts about the user:
{user_details_content}

Use this context naturally in your responses.\
"""

store = InMemoryStore()

# Single shared namespace prefix — used by BOTH nodes
NAMESPACE = ("users", "{user_id}", "details")

# --------------------------------------------------
# Node 1: Remember — extracts and writes new facts
# --------------------------------------------------

def remember_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Extract non-duplicate facts from latest message, write to store."""
    user_id = config["configurable"]["user_id"]
    namespace = ("users", user_id, "details")  # consistent namespace

    items = store.search(namespace)
    user_details_content = (
        "\n".join(f"- {item.value.get('data', '')}" for item in items)
        if items else ""
    )

    last_msg = state["messages"][-1].content

    decision: MemoryDecision = extractor_model.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(user_details_content=user_details_content)),
        HumanMessage(content=last_msg),
    ])

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": mem.text})

    return {}  # no message mutation — write-only node

# --------------------------------------------------
# Node 2: Chat — reads same namespace, responds with context
# --------------------------------------------------

def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Read memory written by remember_node, respond using it."""
    user_id = config["configurable"]["user_id"]
    namespace = ("users", user_id, "details")  # FIXED: matches remember_node

    items = store.search(namespace)
    user_details_content = (
        "\n".join(f"- {item.value.get('data', '')}" for item in items)
        if items else "(empty)"
    )

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details_content)
    )

    response = model.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

# --------------------------------------------------
# Build Graph
# remember (write) → chat (read + respond)
# --------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

# --------------------------------------------------
# Run — last input is deliberate duplicate (dedup test)
# --------------------------------------------------

print("=" * 20 + " WRITE + READ MEMORY DEMO " + "=" * 20)

USER_ID = "u4"
config = {"configurable": {"user_id": USER_ID}}

conversation = [
    "HI! My name is Beast.",
    "HI! I am from INDIA.",
    "I am a Coder.",
    "I am learning LangGraph.",
    "Tell me about short term memory in LLMs.",
    "I am a Coder.",  # duplicate — should not be re-stored
]

for user_input in conversation:
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config)
    print(f"Human:     {user_input}")
    print(f"Assistant: {result['messages'][-1].content}\n")

# --------------------------------------------------
# Final Memory Check — proof of dedup + correct namespace
# --------------------------------------------------

print("=" * 20 + " STORED MEMORY " + "=" * 20)

final_items = store.search(("users", USER_ID, "details"))  # FIXED: matches USER_ID
print(f"Total facts stored: {len(final_items)}\n")

for i, item in enumerate(final_items, 1):
    print(f"[{i}] {item.value['data']}")

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))
