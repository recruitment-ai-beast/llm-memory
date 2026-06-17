"""
LangGraph Token Trimming Demo

Theme: Trimming is NOT deletion.
       Full conversation is stored in checkpointer.
       Only the context WINDOW sent to LLM is trimmed.

Core Insight:
- LLM has a limited context window (token limit)
- Trimming controls what fits IN that window
- Checkpointer stores EVERYTHING regardless
- Think of it as: checkpointer = full library
                  trimmed messages = what you carry in your bag
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
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
# Token Budget
# Max tokens allowed in LLM context window per call
# Everything beyond this is trimmed FROM THE WINDOW
# but remains safely stored in checkpointer
# --------------------------------------------------

MAX_TOKENS = 150

# --------------------------------------------------
# Graph Node — Where Trimming Happens
# --------------------------------------------------

def call_model(state: MessagesState):
    """
    Trims messages to fit token budget before LLM call.
    
    IMPORTANT: trim_messages does NOT delete from state.
    It only controls what gets sent to the LLM this turn.
    Full history remains intact in the checkpointer.
    
    Strategy 'last' = keep most recent messages
    (drops oldest when over token limit)
    """

    # What we ACTUALLY send to LLM (trimmed window)
    trimmed = trim_messages(
        state["messages"],
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS,
        strategy="last"  # keep latest context, drop oldest
    )

    # Diagnostic: show token usage and what LLM sees
    print("\n" + "─" * 50)
    print(f" Total messages in checkpointer: {len(state['messages'])}")
    print(f" Messages sent to LLM (trimmed): {len(trimmed)}")
    print(f" Token count in window: {count_tokens_approximately(messages=trimmed)}/{MAX_TOKENS}")
    print("─" * 50)

    for message in trimmed:
        print(f"  [{type(message).__name__}] {message.content[:60]}...")

    # Invoke LLM with TRIMMED window (not full history)
    response = model.invoke(trimmed)

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
# InMemorySaver — Stores FULL conversation history
# Trimming never touches this — it's the source of truth
# --------------------------------------------------

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

print("\n" + "=" * 20 + " TOKEN TRIMMING DEMO " + "=" * 20)
print("MAX TOKEN WINDOW:", MAX_TOKENS)
print("Watch how trimming controls LLM context")
print("while checkpointer retains full history\n")

config = {"configurable": {"thread_id": "thr-1"}}

# Multi-turn conversation — deliberately exceeds token limit
# to demonstrate trimming in action
user_inputs = [
    [{"role": "user", "content": "HI! I am from INDIA."}],
    [{"role": "user", "content": "I am a Coder."}],
    [{"role": "user", "content": "I am learning LangGraph."}],
    [{"role": "user", "content": "Tell me about short term memory in LLMs."}],
    [{"role": "user", "content": "What are checkpointers?"}],
    # This is the critical turn — will token limit cause forgetting?
    [{"role": "user", "content": "Name my country."}],
]

for msg in user_inputs:
    result = graph.invoke(
        {"messages": msg},
        config=config
    )
    display_latest_turn(result)

# --------------------------------------------------
# Proof: Checkpointer Has FULL History
# Even after trimming, nothing was deleted from state
# This is the core proof of the theme
# --------------------------------------------------

print("\n" + "=" * 20 + " CHECKPOINTER FULL HISTORY " + "=" * 20)
print("Everything below was stored — trimming deleted NOTHING:\n")

full_history = graph.get_state(
    {"configurable": {"thread_id": "thr-1"}}
).values["messages"]

for i, item in enumerate(full_history):
    msg_type = type(item).__name__
    print(f"[{i+1}] {msg_type}: {item.content[:80]}")
    print("─" * 120)

print(f"\n Total messages preserved in checkpointer: {len(full_history)}")
print(" Trimming only controlled the LLM window — never deleted history.")

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))