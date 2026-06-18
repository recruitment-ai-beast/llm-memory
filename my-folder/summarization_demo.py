"""
LangGraph Conversation Summarization Demo

Theme: Summarization is intelligent compression.
       Old messages are permanently deleted —
       their MEANING survives in a summary.

Core Insight:
- Trimming   → drops oldest messages (meaning lost)
- Deletion   → removes permanently (meaning lost)
- Summarization → compresses meaning, then deletes (meaning preserved)

Think of it as:
  Trimming      = tearing out old pages
  Summarization = reading old pages → writing cliff notes → 
                  then tearing them out

Real-world use: Long-running assistants, customer support bots,
                any agent that must remember across many turns
                without exploding token costs.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

# --------------------------------------------------
# Setup
# --------------------------------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=100  # higher than before — summary generation needs more tokens
)

console = Console()

# --------------------------------------------------
# Extended State
# Adds 'summary' field on top of standard MessagesState
# Summary persists in checkpointer alongside messages
# Acts as compressed long-term context
# --------------------------------------------------

class ConversationState(MessagesState):
    summary: str  # rolling summary of deleted messages

# --------------------------------------------------
# Node 1: LLM Response
# Injects existing summary as system context
# before invoking LLM — this is how deleted context
# stays accessible despite permanent message removal
# --------------------------------------------------

def call_model(state: ConversationState):
    """
    Build context-aware message list for LLM.
    
    If a summary exists, prepend it as system message.
    This gives LLM access to compressed history
    even after those messages were permanently deleted.
    """
    context = []

    # Inject summary as system context if it exists
    # This is the key mechanism — deleted messages live on as summary
    if state["summary"]:
        context.append({
            "role": "system",
            "content": f"Summary of conversation so far:\n{state['summary']}"
        })

    # Append live (non-deleted) messages after summary
    context.extend(state["messages"])

    response = model.invoke(context)
    return {"messages": [response]}

# --------------------------------------------------
# Node 2: Summarize + Delete
# Triggered when message count exceeds threshold
# 
# Flow:
# 1. Summarize full conversation (or extend old summary)
# 2. Permanently delete all messages except last 2
# 3. Store new summary in state
#
# Result: Token count resets. Meaning survives.
# --------------------------------------------------

def summarize_and_compress(state: ConversationState):
    """
    Compress conversation history into a rolling summary,
    then permanently delete old messages.

    Why keep last 2?
    - Maintains immediate conversational continuity
    - LLM needs recent turn to respond coherently

    Why delete the rest?
    - Token cost control
    - Old messages are now encoded in summary
    - Permanent deletion — not trimming
    """
    existing_summary = state["summary"]

    # Build summarization prompt
    # If summary exists — extend it rather than rewrite
    if existing_summary:
        summarization_prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend this summary to include the new conversation turns above. "
            "Be concise. Preserve all key facts about the user."
        )
    else:
        summarization_prompt = (
            "Summarize this conversation. "
            "Be concise. Preserve all key facts about the user."
        )

    # Append prompt as human message for LLM to act on
    messages_for_summary = state["messages"] + [
        HumanMessage(content=summarization_prompt)
    ]

    new_summary = model.invoke(messages_for_summary).content

    # Permanently delete all messages except the last 2 turns
    # These are GONE from checkpointer — not trimmed, deleted
    messages_to_delete = state["messages"][:-2]

    print("\n" + "─" * 50)
    print(f" Summary generated — compressing history")
    print(f"  Permanently deleting {len(messages_to_delete)} messages")
    print(f" Retaining last 2 messages for continuity")
    print(f" Summary:\n{new_summary}")
    print("─" * 50)

    return {
        "summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete]
    }

# --------------------------------------------------
# Conditional Router
# Decides whether to summarize after each LLM turn
# Threshold: 6 messages — tune based on token budget
# --------------------------------------------------

def should_summarize(state: ConversationState) -> bool:
    """Trigger summarization when message count exceeds threshold."""
    return len(state["messages"]) > 6

# --------------------------------------------------
# Build Graph
# Flow: LLM responds → check threshold → 
#       summarize if needed → END
# --------------------------------------------------

builder = StateGraph(ConversationState)

builder.add_node("call_model", call_model)
builder.add_node("summarize_and_compress", summarize_and_compress)

builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    should_summarize,
    {
        True: "summarize_and_compress",   # threshold hit → compress
        False: END                         # under threshold → continue
    }
)
builder.add_edge("summarize_and_compress", END)

# --------------------------------------------------
# Helper: Display Latest Turn
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
# Helper: Show Full State
# Proves summarization deleted messages but preserved meaning
# --------------------------------------------------

def show_state(graph, config):
    """
    Inspect checkpointer state after conversation.
    
    Key proof points:
    - summary field contains compressed history
    - message count is low despite long conversation
    - deleted messages are unrecoverable
    """
    snap = graph.get_state(config)
    vals = snap.values

    print("\n" + "=" * 20 + " FINAL STATE " + "=" * 20)
    print(f"""
┌──────────────────┬─────────────────────────────────────┐
│ Field            │ Value                               │
├──────────────────┼─────────────────────────────────────┤
│ Messages in store│ {len(vals.get('messages', []))}                                   │
│ Summary exists   │ {'Yes' if vals.get('summary') else 'No'}                                  │
└──────────────────┴─────────────────────────────────────┘
    """)

    print("Summary (compressed memory):")
    print(vals.get("summary", "None"))

    print("\nRemaining messages (survived deletion):")
    for m in vals.get("messages", []):
        print(f"  [{type(m).__name__}] {m.content[:80]}")

    print("\n Old messages permanently deleted.")
    print(" Meaning preserved in summary.")
    print(" LLM can still reference deleted context via summary injection.")

# --------------------------------------------------
# Run Conversation
# --------------------------------------------------

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thr-a"}}

print("\n" + "=" * 20 + " SUMMARIZATION DEMO " + "=" * 20)
print("Threshold: 6 messages → triggers summarize + delete")
print("Watch how meaning survives even after deletion\n")

def run_turn(text: str):
    """Run one conversation turn."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=text)], "summary": ""},
        config=config
    )
    display_latest_turn(result)

# Deliberately exceeds threshold to trigger summarization
conversation = [
    "HI! I am from INDIA.",
    "I am a Coder.",
    "I am learning LangGraph.",
    "Tell me about short term memory in LLMs.",
    "What are checkpointers?",
    # Critical test — will LLM know after deletion + summary?
    "Name my country.",
]

for turn in conversation:
    run_turn(turn)

# Inspect final state — proof that deletion happened
# but meaning was preserved via summary
show_state(graph, config)

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))