"""Check .env loading and one minimal Groq completion without running CiteGuard."""
from pathlib import Path
import os

from citeguard.llm import DOTENV_PATH, llm_available, _chat

print(f"Project .env path: {DOTENV_PATH}")
print(f".env file exists: {DOTENV_PATH.is_file()}")
print(f"GROQ_API_KEY available to this process: {llm_available()}")
print(f"GROQ_MODEL override: {os.environ.get('GROQ_MODEL') or '(default model order)'}")

if not llm_available():
    print("FAILED: Add GROQ_API_KEY=your_key to the project-root .env file or set it in the shell.")
    raise SystemExit(1)

answer = _chat(
    "Reply with exactly this text and nothing else: CiteGuard Groq connection OK",
    max_tokens=256,
)
if answer:
    print("SUCCESS: A non-empty completion was returned.")
    print(f"Response: {answer}")
else:
    print("FAILED: No usable response. Review the model-specific diagnostics above.")
    raise SystemExit(2)
