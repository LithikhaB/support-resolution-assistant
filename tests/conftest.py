"""Keep automated tests offline unless a test explicitly injects a language client."""

import os

os.environ["LLM_ENABLED"] = "false"
