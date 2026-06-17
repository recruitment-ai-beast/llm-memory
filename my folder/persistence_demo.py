"""
LangGraph Persistent Memory Demo

Theme: Unlike RAM (volatile), database memory is permanent.
       Conversations persist across sessions, threads, and restarts.
       
Key Concept:
- thread-1 knows you're from India (memory persists)
- thread-2 has never met you (isolated memory)
- Restart the runtime — thread-1 still remembers (persistent DB)
"""

import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

# --------------------------------------------------
# Setup
# --------------------------------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=40
)

console = Console()

# --------------------------------------------------
# Graph Node
# --------------------------------------------------

def call_model(state: MessagesState):
    """Invoke LLM with full conversation state."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# --------------------------------------------------
# Build Graph
# --------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# --------------------------------------------------
# Helper
# --------------------------------------------------

def display_latest_turn(result):
    """Display only the latest human/AI exchange."""
    messages = result["messages"]
    human_msg = messages[-2]
    ai_msg = messages[-1]

    console.print("\n[bold green]Human:[/bold green]")
    console.print(Markdown(human_msg.content))
    console.print("\n[bold blue]AI:[/bold blue]")
    console.print(Markdown(ai_msg.content))

# --------------------------------------------------
# Persistent Storage Setup
# Unlike InMemorySaver (RAM) — this survives restarts
# SQLite writes to disk: memory_demo.db
# --------------------------------------------------

conn = sqlite3.connect(
    database="memory_demo.db",
    check_same_thread=False  # required for LangGraph's async access
)

checkpointer = SqliteSaver(conn=conn)

# Compile graph with persistent checkpointer
graph = builder.compile(checkpointer=checkpointer)

# --------------------------------------------------
# Thread 1 — Persistent Conversation
# Same thread_id = same memory across ALL invocations
# This memory survives runtime restarts (unlike RAM)
# --------------------------------------------------

print("\n" + "=" * 15 + " THREAD-1 — PERSISTENT MEMORY " + "=" * 15)

config_thread1 = {"configurable": {"thread_id": "thread-1"}}

# Turn 1: Establish context
result = graph.invoke(
    {"messages": [{"role": "user", "content": "HI! I am from INDIA."}]},
    config=config_thread1
)
display_latest_turn(result)

# Turn 2: Test retrieval — LLM should remember
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Name my country?"}]},
    config=config_thread1
)
display_latest_turn(result)

# --------------------------------------------------
# Thread 2 — Isolated Memory
# Different thread_id = completely separate memory
# Proves memory is scoped, not global
# --------------------------------------------------

print("\n" + "=" * 15 + " THREAD-2 — ISOLATED MEMORY " + "=" * 15)

config_thread2 = {"configurable": {"thread_id": "thread-2"}}

# Thread-2 has NO context about India — never told it
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Name my country?"}]},
    config=config_thread2
)
display_latest_turn(result)

# --------------------------------------------------
# Thread 1 Again — Proving DB Persistence
# Even after thread-2 ran, thread-1 memory is intact
# This is the core proof: DB memory != RAM memory
# --------------------------------------------------

print("\n" + "=" * 15 + " THREAD-1 AGAIN — DB PERSISTENCE PROOF " + "=" * 15)

# Thread-1 still knows — memory written to disk, not RAM
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Name my country?"}]},
    config=config_thread1
)
display_latest_turn(result)

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))