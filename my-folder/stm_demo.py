
"""
LangGraph Memory Demo

Without memory:
- Each invocation is independent.
- The model cannot remember previous messages.

With memory:
- Conversation history is stored using InMemorySaver.
- The model can access previous messages in the same thread.
"""

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from rich.console import Console
from rich.markdown import Markdown

# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=40
)

console = Console()

# --------------------------------------------------
# Graph Node
# --------------------------------------------------

def call_model(state: MessagesState):
    """
    Receives conversation state and returns
    the model's response.
    """
    response = model.invoke(state["messages"])

    return {
        "messages": [response]
    }

# --------------------------------------------------
# Build Graph
# --------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def display_latest_turn(result):
    """
    Prints only the latest user message
    and latest AI response.
    """
    messages = result["messages"]

    human_msg = messages[-2]
    ai_msg = messages[-1]

    console.print("\n[bold green]Human:[/bold green]")
    console.print(Markdown(human_msg.content))

    console.print("\n[bold blue]AI:[/bold blue]")
    console.print(Markdown(ai_msg.content))

# ==================================================
# WITHOUT MEMORY
# ==================================================

print("\n" + "=" * 15 + " WITHOUT MEMORY " + "=" * 15)

graph = builder.compile()

user_inputs = [
    [{"role": "user", "content": "HI! I am from INDIA."}],
    [{"role": "user", "content": "Name my country?"}]
]

for msg in user_inputs:
    result = graph.invoke({"messages": msg})
    display_latest_turn(result)

# ==================================================
# WITH MEMORY
# ==================================================

print("\n" + "=" * 15 + " WITH MEMORY " + "=" * 15)

memory = InMemorySaver()

graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

user_inputs = [
    [{"role": "user", "content": "HI! I am from INDIA."}],
    [{"role": "user", "content": "Name my country?"}]
]

for msg in user_inputs:
    result = graph.invoke(
        {"messages": msg},
        config=config
    )

    display_latest_turn(result)

# ==================================================
# WORKFLOW GRAPH 
# ==================================================
from IPython.display import display,Image
display(Image(graph.get_graph().draw_mermaid_png())) 