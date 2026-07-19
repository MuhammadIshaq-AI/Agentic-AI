"""Writer + Critic reflection loop built with LangGraph + Ollama (Gemma).

The graph is a CYCLE — the thing plain LangChain chains can't do:

        ┌──────────┐        ┌──────────┐
        │  write   │ ─────▶ │ critique │
        └──────────┘        └────┬─────┘
             ▲                   │
             │   score < threshold & revisions left
             └───────────────────┘
                                 │ score >= threshold OR out of revisions
                                 ▼
                              (finish)

Gemma has weak/no native tool-calling, so we DON'T use `.bind_tools()`.
Instead the critic returns structured JSON (`format="json"`) that we parse,
and LangGraph's conditional edge routes on it. This is the portable pattern
for local models without solid tool-calling.
"""

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

import config


# ---- Shared state passed between nodes -------------------------------------
class State(TypedDict):
    task: str          # the original writing request
    draft: str         # latest draft
    critique: str      # latest critique text
    score: int         # latest quality score (0-10)
    revisions: int     # how many revision rounds we've done


# ---- Two LLM handles: one free-form (writer), one JSON-only (critic) -------
writer_llm = ChatOllama(
    model=config.MODEL, base_url=config.BASE_URL, temperature=0.7
)
critic_llm = ChatOllama(
    model=config.MODEL, base_url=config.BASE_URL,
    temperature=config.TEMPERATURE, format="json",
)


# ---- Nodes -----------------------------------------------------------------
def write_node(state: State) -> dict:
    """Produce (or revise) a draft."""
    if state.get("draft"):
        # Revision pass — incorporate the critic's feedback.
        prompt = (
            f"Task: {state['task']}\n\n"
            f"Your previous draft:\n{state['draft']}\n\n"
            f"Editor feedback:\n{state['critique']}\n\n"
            "Rewrite the draft to address every point of feedback. "
            "Return ONLY the improved draft."
        )
    else:
        prompt = f"Write a first draft for this task:\n{state['task']}"

    msg = writer_llm.invoke([
        SystemMessage(content="You are a skilled writer. Write clearly and concisely."),
        HumanMessage(content=prompt),
    ])
    print(f"\n--- DRAFT (revision {state.get('revisions', 0)}) ---\n{msg.content}\n")
    return {"draft": msg.content}


def critique_node(state: State) -> dict:
    """Score the draft and give feedback — as parseable JSON."""
    prompt = (
        f"Task the writer was given:\n{state['task']}\n\n"
        f"Draft to evaluate:\n{state['draft']}\n\n"
        "Evaluate the draft. Respond with a JSON object shaped EXACTLY like:\n"
        '{"score": <integer 0-10>, "feedback": "<specific, actionable feedback>"}'
    )
    msg = critic_llm.invoke([
        SystemMessage(content="You are a demanding editor. Output only valid JSON."),
        HumanMessage(content=prompt),
    ])

    # Robust parse — tiny models sometimes wrap or mangle JSON.
    try:
        data = json.loads(msg.content)
        score = int(data.get("score", 0))
        feedback = str(data.get("feedback", "")).strip()
    except (json.JSONDecodeError, ValueError, TypeError):
        score, feedback = 0, f"(Could not parse critic output: {msg.content!r})"

    print(f"--- CRITIQUE --- score={score}/10\n{feedback}\n")
    return {
        "score": score,
        "critique": feedback,
        "revisions": state.get("revisions", 0) + 1,
    }


def should_continue(state: State) -> str:
    """Conditional edge: loop back to the writer, or stop."""
    if state["score"] >= config.QUALITY_THRESHOLD:
        print(f"[ok] score {state['score']} >= {config.QUALITY_THRESHOLD} — done.")
        return "finish"
    if state["revisions"] >= config.MAX_REVISIONS:
        print(f"[stop] hit MAX_REVISIONS ({config.MAX_REVISIONS}) — done.")
        return "finish"
    print("[loop] revising...")
    return "revise"


# ---- Wire the graph --------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("write", write_node)
    g.add_node("critique", critique_node)

    g.add_edge(START, "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges(
        "critique",
        should_continue,
        {"revise": "write", "finish": END},
    )
    return g.compile()
