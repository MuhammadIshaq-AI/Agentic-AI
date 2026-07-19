"""Central config — change your Ollama model tag here in ONE place.

Your installed model is `gemma3:270m` (tiny, ~270M params). It runs fast but
its reasoning/JSON reliability is weak. For noticeably better agent behaviour:
    ollama pull gemma3:1b     # ~800 MB
    ollama pull gemma3:4b     # ~3.3 GB
then update MODEL below.
"""

# The Ollama model tag (see `ollama list`).
MODEL = "gemma3:270m"

# Ollama server URL (default local install).
BASE_URL = "http://localhost:11434"

# Lower = more deterministic. Good for critique/scoring.
TEMPERATURE = 0.3

# Stop looping once the critic's score reaches this (out of 10),
# or once MAX_REVISIONS is hit — whichever comes first.
QUALITY_THRESHOLD = 8
MAX_REVISIONS = 3
