"""Run the Writer + Critic reflection loop.

Usage:
    python main.py
    python main.py "Write a haiku about local LLMs"
"""

import sys

from graph import build_graph


def main() -> None:
    task = (
        " ".join(sys.argv[1:])
        or "Write a short, punchy paragraph explaining what an AI agent is "
           "to a non-technical reader."
    )

    app = build_graph()
    print(f"=== TASK ===\n{task}\n")

    final = app.invoke({"task": task, "revisions": 0})

    print("=" * 60)
    print("FINAL DRAFT:\n")
    print(final["draft"])
    print(f"\n(final score: {final['score']}/10 after {final['revisions']} revision(s))")


if __name__ == "__main__":
    main()
