"""Reliable Groq integration for CiteGuard.

Loads the project-root .env file, checks for genuinely usable completions,
falls through when a model returns empty content, and parses support judgments
without assuming every response is perfectly formatted.
"""

import os
import re
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Try python-dotenv first. A tiny fallback loader keeps the app functional if
# python-dotenv has not been installed yet. Existing process env vars take
# precedence over .env values.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
except ImportError:
    if DOTENV_PATH.is_file():
        try:
            for raw_line in DOTENV_PATH.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value and value[0:1] in ("'", '"') and value[-1:] == value[0:1]:
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError as exc:
            print(f"Could not read project .env file: {exc}")

# The first model is fast. The next two are fallback options if the first
# returns empty content or is unavailable to this account.
MODEL_CANDIDATES = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

_working_model = None
_CONNECT_TIMEOUT = 4
_READ_TIMEOUT = 18


def llm_available() -> bool:
    """Return whether an API key is configured, not whether the API is live."""
    return bool((os.environ.get("GROQ_API_KEY") or "").strip())


def _model_order():
    models = list(MODEL_CANDIDATES)
    configured_model = (os.environ.get("GROQ_MODEL") or "").strip()
    if configured_model:
        models = [configured_model] + [m for m in models if m != configured_model]
    if _working_model and _working_model in models:
        models.remove(_working_model)
        models.insert(0, _working_model)
    return models


def _payload_for_model(model_name, prompt, max_tokens):
    system_message = (
        "You are CiteGuard, an assistant for scientific manuscript review. "
        "Follow the requested output format exactly. Be concise, evidence-based, "
        "and do not invent paper details or evidence."
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        # Groq documents max_completion_tokens as the replacement for max_tokens.
        "max_completion_tokens": max(256, int(max_tokens)),
        "temperature": 0.1,
    }

    # Reasoning tokens can consume the completion budget before a visible final
    # answer is produced. Set low effort for GPT-OSS; disable it for Qwen 3.8.
    if model_name.startswith("openai/gpt-oss-"):
        payload["reasoning_effort"] = "low"
        payload["include_reasoning"] = False
    elif model_name == "qwen/qwen3.8-27b":
        payload["reasoning_effort"] = "none"

    return payload


def _chat(prompt, max_tokens=512) -> Optional[str]:
    """Call Groq and return non-empty visible answer text, or None on failure.

    Empty 200 responses are not considered success: the next configured model
    is tried instead. API keys are never included in logs.
    """
    global _working_model

    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        print(
            "Groq LLM is not configured. Expected GROQ_API_KEY in the process "
            f"environment or {DOTENV_PATH}."
        )
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for model_name in _model_order():
        payload = _payload_for_model(model_name, prompt, max_tokens)
        try:
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            if response.status_code in (401, 403):
                print(
                    f"Groq authentication/permission error ({response.status_code}); "
                    "check that GROQ_API_KEY is valid and has model access."
                )
                return None
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                print(f"Groq returned no choices from {model_name}; trying fallback.")
                continue

            choice = choices[0] or {}
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                # Be tolerant of OpenAI-compatible APIs returning content parts.
                content = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            content = content.strip() if isinstance(content, str) else ""

            if not content:
                finish_reason = choice.get("finish_reason") or "unknown"
                usage = data.get("usage") or {}
                completion_tokens = usage.get("completion_tokens", "unknown")
                print(
                    f"Groq returned empty visible content from {model_name} "
                    f"(finish_reason={finish_reason}, completion_tokens={completion_tokens}); "
                    "trying fallback."
                )
                continue

            _working_model = model_name
            return content

        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else "unknown"
            body = ""
            if response is not None:
                body = (response.text or "")[:240].replace("\n", " ")
            print(f"Groq request failed for {model_name} (HTTP {status}): {body}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            print(f"Groq connection/timeout for {model_name}: {exc.__class__.__name__}; trying fallback.")
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            print(f"Groq response failed for {model_name}: {exc.__class__.__name__}; trying fallback.")

    print("All configured Groq models failed or returned empty content.")
    return None


def judge_support(sentence, abstract):
    prompt = (
        "Decide whether the paper abstract provides evidence that supports the "
        "specific claim in the sentence. Topic similarity alone is not support. "
        "If the abstract does not support the claim, answer NO. Do not use outside knowledge.\n\n"
        f"CLAIM: {sentence}\n\n"
        f"PAPER ABSTRACT: {abstract}\n\n"
        "Output exactly two lines. First line: YES or NO. Second line: one concise reason."
    )
    reply = _chat(prompt, max_tokens=768)
    if not reply:
        return {"supported": None, "reason": "Groq returned no usable completion from the configured models."}

    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return {"supported": None, "reason": "Groq returned an empty response."}

    # Accept plain YES/NO and common Markdown/bullet formatting, but never guess
    # a verdict if the answer does not clearly begin with YES or NO.
    first = re.sub(r"^[\s>*#`]+", "", lines[0]).strip()
    verdict_match = re.match(r"^(YES|NO)\b", first, flags=re.IGNORECASE)
    if not verdict_match:
        return {
            "supported": None,
            "reason": f"Groq returned an unparseable verdict: {reply[:300]}",
        }

    supported = verdict_match.group(1).upper() == "YES"
    reason = " ".join(lines[1:]).strip() or first[len(verdict_match.group(1)):].strip(" :.-")
    if not reason:
        reason = "The model returned a verdict without an explanation."
    return {"supported": supported, "reason": reason}


def suggest_for_claim(sentence, candidates):
    if not candidates:
        return "No candidate source abstracts were retrieved for this claim."

    listed = "\n\n".join(
        f"Candidate {i + 1}: {c.get('title', 'Untitled paper')}\n"
        f"Abstract/excerpt: {str(c.get('text', c.get('abstract', '')) or '')[:700]}"
        for i, c in enumerate(candidates[:4])
    )
    prompt = (
        "You are a conservative scientific evidence-review assistant.\n\n"
        "CLAIM TO INVESTIGATE:\n"
        f"{sentence}\n\n"
        "RETRIEVED CANDIDATE PAPERS:\n"
        f"{listed}\n\n"
        "For each candidate, classify its relationship to the claim as "
        "DIRECT SUPPORT, PARTIAL SUPPORT, or NO SUPPORT.\n\n"
        "STRICT RULES:\n"
        "1. Judge only the title and abstract provided. Do not invent evidence.\n"
        "2. DIRECT SUPPORT means the available text supports the important "
        "parts of the claim, including its scope and strength.\n"
        "3. PARTIAL SUPPORT means the source discusses the topic or supports "
        "a weaker version of the claim, but does not establish the whole claim.\n"
        "4. NO SUPPORT means the supplied text provides no relevant evidence.\n"
        "5. Words such as 'always', 'every', 'regardless', and 'eliminates', "
        "as well as exact percentages, require explicit evidence. A general "
        "improvement does not establish a claimed percentage or universal rule.\n"
        "6. Never describe a source as directly supporting a claim merely "
        "because it discusses the same model, technique, or research area.\n"
        "7. Quote a short passage from the supplied text when available. "
        "If the text does not establish the claim, say what is missing.\n"
        "8. If no candidate directly supports the complete claim, state this "
        "explicitly. You may still recommend a partially relevant source.\n\n"
        "Return concise results under these headings:\n"
        "Candidate and title:\n"
        "Classification:\n"
        "Evidence from supplied text:\n"
        "Limitation:\n"
        "Overall recommendation:\n"
    )
    reply = _chat(prompt, max_tokens=640)
    if reply:
        return reply
    return None
