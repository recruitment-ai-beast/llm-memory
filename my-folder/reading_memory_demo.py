"""
LangGraph Long-Term Memory Demo

Theme: Chatbot reads pre-existing user memory before responding.
       Memory Store acts as a persistent user profile —
       injected into every LLM call as system context.

Checkpointer  → what was SAID    (conversation thread)
Memory Store  → what is KNOWN    (user profile, pre-loaded)
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
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
# System Prompt Template
# User facts from Memory Store are injected here
# before every LLM call
# --------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant with knowledge about the user.

Known facts about the user:
{user_facts}

Use this context naturally in your responses.\
"""

# --------------------------------------------------
# Pre-load User Memory
# Simulates an existing user profile in Memory Store
# In production: populated over time from past sessions
# --------------------------------------------------

store = InMemoryStore()

USER_ID = "u1"
NAMESPACE = ("users", USER_ID, "details")

user_profile = [
    "User's name is Beast.",
    "User is from India.",
    "User is a coder.",
    "User is learning LangGraph.",
]

for idx, fact in enumerate(user_profile, 1):
    store.put(NAMESPACE, str(idx), {"data": fact})

# --------------------------------------------------
# Graph Node
# Reads Memory Store → builds system prompt → 
# injects into LLM context
# --------------------------------------------------

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Fetch user profile from Memory Store.
    Inject as system context before LLM call.
    LLM responds as if it already knows the user.
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("users", user_id, "details")

    # Retrieve all stored facts for this user
    items = store.search(namespace)

    user_facts = (
        "\n".join(f"- {item.value.get('data', '')}" for item in items)
        if items else "No user profile found."
    )

    # Build context-aware message list
    system_message = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(user_facts=user_facts)
    )

    messages = [system_message] + state["messages"]

    response = model.invoke(messages)
    return {"messages": [response]}

# --------------------------------------------------
# Build Graph
# --------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Pass store= so nodes can access Memory Store
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer, store=store)

config = {"configurable": {"thread_id": "t1", "user_id": USER_ID}}

# --------------------------------------------------
# Run — LLM should answer using pre-loaded memory
# --------------------------------------------------

print("\n" + "=" * 20 + " MEMORY-AWARE CHATBOT " + "=" * 20)

queries = [
    "What is my name?",
    "Where am I from?",
    "Explain gen ai in simple terms.",
    "Name my country.",
]

for query in queries:
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config
    )
    console.print(f"\n[bold green]Human:[/bold green] {query}")
    console.print(f"[bold blue]AI:[/bold blue] {result['messages'][-1].content}")

# --------------------------------------------------
# Graph Visualization
# --------------------------------------------------

from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))