"""Groq chat-completions client with empty-response and reasoning-budget handling."""
import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_CANDIDATES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]
_working_model = None


def llm_available():
    """Return whether an API key is configured, not whether the API is reachable."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def _content_to_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Accommodate APIs that return content as typed text segments.
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _chat(prompt, max_tokens=256):
    """Call Groq and try the next model if a request fails or returns empty content."""
    global _working_model

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    models = (
        [_working_model] + [m for m in MODEL_CANDIDATES if m != _working_model]
        if _working_model
        else list(MODEL_CANDIDATES)
    )

    # GPT-OSS spends completion tokens on reasoning as well as the visible answer.
    # A 100-120 token budget can be consumed before it emits message.content.
    completion_budget = max(768, int(max_tokens) * 4)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for model_name in models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": completion_budget,
            "temperature": 0.2,
            "stream": False,
        }
        if model_name.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"
            payload["reasoning_format"] = "hidden"
        elif model_name.startswith("qwen/qwen3"):
            payload["reasoning_effort"] = "none"

        try:
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                print(f"LLM returned no choices from {model_name}; trying fallback model.")
                continue

            choice = choices[0]
            message = choice.get("message") or {}
            content = _content_to_text(message.get("content"))
            if not content:
                finish_reason = choice.get("finish_reason", "unknown")
                reasoning_chars = len(str(message.get("reasoning") or ""))
                print(
                    f"LLM returned empty content from {model_name} "
                    f"(finish_reason={finish_reason}, reasoning_chars={reasoning_chars}); "
                    "trying fallback model."
                )
                continue

            _working_model = model_name
            return content

        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            print(f"LLM call failed with {model_name}: {exc}; trying fallback model.")

    return None


def judge_support(sentence, abstract):
    prompt = (
        "Decide whether the paper abstract provides evidence supporting the claim.\n"
        "Reply with YES or NO as the first line, followed by one short reason. "
        "Do not infer support from topic similarity alone. If the abstract does not "
        "contain enough evidence, answer NO.\n\n"
        f"Claim: {sentence}\n\n"
        f"Paper abstract: {abstract}\n"
    )
    reply = _chat(prompt, max_tokens=180)
    if not reply:
        return {
            "supported": None,
            "reason": "LLM unavailable or returned an empty response; embedding fallback used.",
        }

    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return {"supported": None, "reason": "LLM returned no parseable response."}

    first_line = lines[0].upper().strip(" .:*_`-\t")
    if first_line.startswith("YES"):
        supported = True
    elif first_line.startswith("NO"):
        supported = False
    else:
        # Handle a verdict that appears after a short prefix, without silently guessing.
        normalized = " ".join(lines[0].upper().split())
        if normalized.startswith("SUPPORTED"):
            supported = True
        elif normalized.startswith("NOT SUPPORTED") or normalized.startswith("UNSUPPORTED"):
            supported = False
        else:
            return {
                "supported": None,
                "reason": f"LLM returned an unrecognized verdict: {lines[0]}",
            }

    reason = lines[1] if len(lines) > 1 else "No explanation supplied."
    return {"supported": supported, "reason": reason}


def suggest_for_claim(sentence, candidates):
    if not candidates:
        return "No candidate papers were retrieved, so no source suggestion is available."

    listed_items = []
    for candidate in candidates:
        title = str(candidate.get("title") or "Untitled paper")
        text = str(candidate.get("text") or candidate.get("abstract") or "")
        listed_items.append(f"- {title}: {text[:250]}")

    prompt = (
        "A manuscript sentence has no citation. Consider only the candidate papers below.\n"
        "If one plausibly supports the sentence, name it and explain the connection in one sentence. "
        "Do not claim that a paper supports the sentence based on its title alone. "
        "If none is suitable, say that none of the candidates clearly supports it.\n\n"
        f"Sentence: {sentence}\n\n"
        "Candidate papers:\n"
        + "\n".join(listed_items)
    )
    reply = _chat(prompt, max_tokens=220)
    if not reply:
        return "No suggestion available: all configured LLM models failed or returned empty content."
    return reply.strip()
