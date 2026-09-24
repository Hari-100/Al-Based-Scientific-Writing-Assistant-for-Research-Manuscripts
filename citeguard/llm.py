import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_CANDIDATES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

_working_model = None


def llm_available():
    return bool(os.environ.get("GROQ_API_KEY"))


def _chat(prompt, max_tokens=120):
    global _working_model

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Try the model that worked last time first.
    models = (
        [_working_model]
        + [m for m in MODEL_CANDIDATES if m != _working_model]
        if _working_model
        else MODEL_CANDIDATES
    )

    for model_name in models:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            r = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=15,
            )

            r.raise_for_status()

            data = r.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )

            if not content or not content.strip():
                print(f"llm returned empty response from {model_name}")
                continue

            content = content.strip()

            _working_model = model_name
            return content

        except requests.RequestException as e:
            print(f"llm call failed with {model_name}: {e}")

        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"unexpected llm response from {model_name}: {e}")

    return None


def judge_support(sentence, abstract):
    prompt = (
        "Sentence: " + sentence + "\n"
        "Paper abstract: " + abstract + "\n"
        "Does the abstract support the claim made in the sentence?\n"
        "Answer YES or NO on the first line, then a one sentence reason "
        "on the next line."
    )

    reply = _chat(prompt)

    if not reply or not reply.strip():
        return {
            "supported": None,
            "reason": "llm returned an empty response",
        }

    lines = reply.strip().splitlines()

    if not lines:
        return {
            "supported": None,
            "reason": "llm returned no parseable response",
        }

    first_line = lines[0].strip().upper()

    if first_line.startswith("YES"):
        supported = True
    elif first_line.startswith("NO"):
        supported = False
    else:
        return {
            "supported": None,
            "reason": f"llm returned an unexpected verdict: {reply}",
        }

    reason = lines[1].strip() if len(lines) > 1 else reply

    return {
        "supported": supported,
        "reason": reason,
    }


def suggest_for_claim(sentence, candidates):
    listed = "\n".join(
        f"- {c['title']}: {c['text'][:200]}"
        for c in candidates
    )

    prompt = (
        "An author wrote this sentence without a citation: "
        + sentence
        + "\n"
        "Candidate papers already verified by the system:\n"
        + listed
        + "\n"
        "If one of these genuinely supports the sentence, name it and "
        "explain why in one sentence. If none fit well, say so plainly."
    )

    return _chat(prompt, max_tokens=100)