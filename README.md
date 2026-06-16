# LangGraph Memory Demo

Demonstrates the difference between stateless and stateful 
LLM conversations using LangGraph's checkpoint system.

---

## What This Shows

| Mode | Behaviour |
|------|-----------|
| Without Memory | Each message is independent — no context retained |
| With Memory | Full conversation history persists across turns |

---

## The Core Concept

Without memory, ask an LLM "what's my country?" after 
telling it you're from India — it has no idea.

With `InMemorySaver`, the graph checkpoints every turn. 
Same question, same thread — it remembers.

---

## Stack

- **LangGraph** — stateful graph execution
- **InMemorySaver** — in-memory conversation checkpointing
- **Groq** — LLM inference (`llama-3.1-8b-instant`)
- **Rich** — terminal output formatting

---

## Run

```bash
pip install -r requirements.txt
python stm_demo.py
```

---

## Key Insight

LLM memory is not magic — it's previous messages 
injected back into context via a checkpointer.

Same model. Different state management. 
Completely different behaviour.

---

## Part of

**BEAST — Vertical AI Engineer**  
Building production AI systems for recruitment automation.  
[Twitter](https://x.com/beastbuildsai)
[LinkedIn](https://www.linkedin.com/in/beast-builds-ai) • 
[Live Product](https://ai-for-resume-screening.streamlit.app/)