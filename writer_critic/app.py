"""Streamlit frontend for the Writer + Critic reflection loop.

Run from this folder:
    streamlit run app.py

It streams each node's output live so you can watch the writer draft,
the critic score, and the graph loop or finish.
"""

import config
from graph import build_graph

import streamlit as st

st.set_page_config(page_title="Writer + Critic Agent", page_icon="✍️", layout="centered")

st.title("✍️ Writer + Critic")
st.caption(
    f"LangGraph reflection loop on local Ollama · model **{config.MODEL}**"
)

# --- Sidebar: live-tunable loop settings ------------------------------------
with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** `{config.MODEL}`")
    threshold = st.slider(
        "Quality threshold", 1, 10, config.QUALITY_THRESHOLD,
        help="Loop stops once the critic's score reaches this.",
    )
    max_revisions = st.slider(
        "Max revisions", 1, 6, config.MAX_REVISIONS,
        help="Hard cap on revision rounds.",
    )
    st.markdown("---")
    st.caption(
        "Tip: set threshold to 10 to force the loop to iterate "
        "(a tiny model rarely scores a perfect 10)."
    )

# --- Main input -------------------------------------------------------------
task = st.text_area(
    "Writing task",
    value="Write a short, punchy paragraph explaining what an AI agent is "
          "to a non-technical reader.",
    height=100,
)

run = st.button("Run agent", type="primary")

if run:
    if not task.strip():
        st.warning("Enter a writing task first.")
        st.stop()

    # Apply the sidebar overrides to the shared config for this run.
    config.QUALITY_THRESHOLD = threshold
    config.MAX_REVISIONS = max_revisions

    app = build_graph()

    st.markdown("### Progress")
    progress = st.container()
    final_state = None

    # Stream node-by-node so the user watches the loop unfold.
    with st.spinner("Agent working... (local model — first token can be slow)"):
        for event in app.stream({"task": task, "revisions": 0}):
            for node_name, node_out in event.items():
                if node_name == "write":
                    rev = node_out.get("revisions", 0)
                    with progress.expander(f"📝 Draft (revision {rev})", expanded=True):
                        st.write(node_out["draft"])
                elif node_name == "critique":
                    score = node_out.get("score", 0)
                    with progress.expander(
                        f"🧐 Critique — score {score}/10", expanded=True
                    ):
                        st.progress(score / 10)
                        st.write(node_out.get("critique", ""))
                final_state = {**(final_state or {}), **node_out}

    st.markdown("---")
    st.markdown("### ✅ Final draft")
    st.success(final_state.get("draft", "(no draft produced)"))
    st.caption(
        f"Final score {final_state.get('score', 0)}/10 · "
        f"{final_state.get('revisions', 0)} revision(s)"
    )
