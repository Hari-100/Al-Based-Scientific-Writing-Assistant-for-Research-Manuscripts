import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# tried in order, first one that actually works gets cached and reused.
# check https://console.groq.com/docs/models for the current free model list
# if all of these ever stop working
MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "minimaxai/minimax-m2.7",
]

_working_model = None

# api key: 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
def llm_available():
    return bool(os.environ.get("GROQ_API_KEY"))


def _chat(prompt, max_tokens=120):
    global _working_model

    api_key = os.environ.get("GROQ_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # try the model that worked last time first, then fall through the rest
    models = [_working_model] + [m for m in MODEL_CANDIDATES if m != _working_model] \
        if _working_model else MODEL_CANDIDATES

    for model_name in models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            _working_model = model_name
            return content
        except requests.RequestException as e:
            print(f"llm call failed with {model_name}:", e)

    return None


def judge_support(sentence, abstract):
    prompt = (
        "Sentence: " + sentence + "\n"
        "Paper abstract: " + abstract + "\n"
        "Does the abstract support the claim made in the sentence? "
        "Answer YES or NO on the first line, then a one sentence reason on the next line."
    )
    reply = _chat(prompt)
    if reply is None:
        return {"supported": None, "reason": "llm call failed"}
    lines = reply.strip().splitlines()
    supported = lines[0].strip().upper().startswith("YES")
    reason = lines[1].strip() if len(lines) > 1 else reply
    return {"supported": supported, "reason": reason}


def suggest_for_claim(sentence, candidates):
    listed = "\n".join(f"- {c['title']}: {c['text'][:200]}" for c in candidates)
    prompt = (
        "An author wrote this sentence without a citation: " + sentence + "\n"
        "Candidate papers already verified by the system:\n" + listed + "\n"
        "If one of these genuinely supports the sentence, name it and explain why in one sentence. "
        "If none fit well, say so plainly."
    )
    return _chat(prompt, max_tokens=100)