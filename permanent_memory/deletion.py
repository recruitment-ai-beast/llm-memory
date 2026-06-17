"""
LangGraph Permanent Message Deletion Demo

Theme: Unlike trimming (windowing), deletion is permanent.
       RemoveMessage surgically removes messages from 
       checkpointer state — they are gone forever.

Core Insight:
- Deletion  = removes from checkpointer (permanent loss)

Use case: Privacy compliance, token cost management,
          removing irrelevant context permanently.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

# --------------------------------------------------
# Setup
# --------------------------------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=40
)

console = Console()

# --------------------------------------------------
# Node 1: LLM Call
# Responds BEFORE deletion happens
# So LLM always sees full context on current turn
# --------------------------------------------------

def call_model(state: MessagesState):
    """Invoke LLM with current full message state."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# --------------------------------------------------
# Node 2: Permanent Deletion
# Runs AFTER LLM responds
# Surgically removes oldest messages from checkpointer
# No recovery. No undo. Permanent.
# --------------------------------------------------

def delete_old_messages(state: MessagesState):
    """
    Permanently delete oldest messages when history
    exceeds 10 messages.

    Why 10? Balances context retention vs storage cost.
    Why delete 6? Aggressive cleanup — keeps last 4 + new.

    PERMANENT: RemoveMessage
    modifies the checkpointer state directly.
    These messages cannot be retrieved after deletion.
    """
    messages = state["messages"]

    print("\n" + "─" * 50)
    print(f" Messages in checkpointer BEFORE deletion: {len(messages)}")

    # Trigger deletion only when threshold exceeded
    if len(messages) > 10:
        # Select oldest 6 messages for permanent removal
        to_delete = messages[:6]

        print(f"  Permanently deleting {len(to_delete)} oldest messages:")
        for msg in to_delete:
            print(f"    [{type(msg).__name__}] {msg.content[:60]}")

        print(f" Messages remaining after deletion: {len(messages) - len(to_delete)}")
        print("─" * 50)
        print("  These messages are GONE —  DELETED PERMANENTLY.")

        # RemoveMessage targets by ID — surgical permanent removal
        return {"messages": [RemoveMessage(id=m.id) for m in to_delete]}

    print(f" Under threshold — no deletion triggered")
    print("─" * 50)

    return {}

# --------------------------------------------------
# Build Graph
# Flow: LLM responds first → then delete old messages
# This ensures LLM always has context before cleanup
# --------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("delete_old_messages", delete_old_messages)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "delete_old_messages")
builder.add_edge("delete_old_messages", END)

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
# InMemorySaver — Source of truth for message state
# RemoveMessage will permanently modify THIS store
# --------------------------------------------------

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

print("\n" + "=" * 20 + " PERMANENT DELETION DEMO " + "=" * 20)
print("Deletion threshold: 10 messages")
print("Messages deleted per cleanup: 6 oldest")

config = {"configurable": {"thread_id": "thr-2"}}

# Multi-turn conversation to accumulate message history
# Watch deletion trigger once we exceed 10 messages
user_inputs = [
    [{"role": "user", "content": "HI! I am from INDIA."}],
    [{"role": "user", "content": "I am a Coder."}],
    [{"role": "user", "content": "I am learning LangGraph."}],
    [{"role": "user", "content": "Tell me about short term memory in LLMs."}],
    [{"role": "user", "content": "What are checkpointers?"}],
    # By this turn — deletion may have triggered
    # Watch if LLM still knows country name.
    [{"role": "user", "content": "Name my country."}],
]

for msg in user_inputs:
    result = graph.invoke(
        {"messages": msg},
        config=config
    )
    display_latest_turn(result)

# --------------------------------------------------
# Final Proof: Deletion is Permanent
# Compare this count to total messages sent
# Missing messages = permanently deleted, not trimmed
# --------------------------------------------------

remaining = graph.get_state(config).values["messages"]

print("\n" + "=" * 20 + " DELETION PROOF " + "=" * 20)
print(f"Total messages sent:      {len(user_inputs) * 2}")
print(f"Messages in checkpointer: {len(remaining)}")
print(f"Permanently deleted:      {(len(user_inputs) * 2) - len(remaining)}")
print("\nRemaining messages (what survived deletion):")

for i, msg in enumerate(remaining):
    print(f"  [{i+1}] {type(msg).__name__}: {msg.content[:60]}")

print("\n  Deleted messages are unrecoverable.")

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))