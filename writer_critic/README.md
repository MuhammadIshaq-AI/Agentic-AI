# Writer + Critic

An agentic **reflection loop** built with [LangGraph](https://langchain-ai.github.io/langgraph/) and a local LLM via [Ollama](https://ollama.com/). A *writer* node drafts, a *critic* node scores and gives feedback, and the graph loops until the score clears a threshold or a revision cap is hit — a cyclic, stateful workflow that a linear chain can't express.

```
 START ─▶ write ─▶ critique ─┬─ score < threshold & revisions left ─▶ write
                             └─ score ≥ threshold OR revisions exhausted ─▶ END
```

## Design note: local-model compatibility

Small local models (e.g. Gemma) have weak or absent native tool-calling, so this project avoids `.bind_tools()`. The critic instead emits **structured JSON** (`format="json"`), which is parsed and used to drive a LangGraph conditional edge. The pattern works on any Ollama model regardless of tool-calling support.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with a model pulled:
  ```bash
  ollama pull gemma3:1b   # set the tag you use in config.py
  ```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Web UI** (recommended — streams each step live):
```bash
streamlit run app.py        # http://localhost:8501
```

**CLI:**
```bash
python main.py                                  # default task
python main.py "Write a haiku about local LLMs" # custom task
```

## Configuration

All settings live in `config.py`:

| Setting | Description |
| --- | --- |
| `MODEL` | Ollama model tag (must match `ollama list`) |
| `BASE_URL` | Ollama server URL (default `http://localhost:11434`) |
| `QUALITY_THRESHOLD` | Score (0–10) that ends the loop |
| `MAX_REVISIONS` | Hard cap on revision rounds |

The Streamlit sidebar can override `QUALITY_THRESHOLD` and `MAX_REVISIONS` per run. Set the threshold to `10` to force the loop to iterate.

## Project layout

| File | Responsibility |
| --- | --- |
| `config.py` | Model tag and loop settings |
| `graph.py` | State schema, writer/critic nodes, cyclic graph wiring |
| `main.py` | CLI entry point |
| `app.py` | Streamlit frontend (streams nodes as they run) |

## Concepts demonstrated

- **Cyclic graphs** — feedback loops beyond linear chains
- **Conditional edges** — routing on parsed model output
- **Shared typed state** — passed across nodes
- **Structured-output routing** — tool-calling workaround for local models
