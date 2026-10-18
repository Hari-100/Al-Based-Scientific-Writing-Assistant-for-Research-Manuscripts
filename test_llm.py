import os
import unittest
from unittest.mock import patch

from citeguard import llm


class FakeResponse:
    def __init__(self, data=None, status_code=200, text=""):
        self._data = data or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self._data


def completion(content, finish_reason="stop"):
    return FakeResponse({
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"completion_tokens": 32},
    })


class LLMTests(unittest.TestCase):
    def setUp(self):
        llm._working_model = None

    def tearDown(self):
        llm._working_model = None

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("citeguard.llm.requests.post")
    def test_empty_first_response_falls_through(self, post):
        post.side_effect = [completion(""), completion("YES\nThe abstract supports this.")]
        answer = llm._chat("test prompt", max_tokens=256)
        self.assertEqual(answer, "YES\nThe abstract supports this.")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["max_completion_tokens"], 256)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("citeguard.llm.requests.post")
    def test_empty_responses_do_not_count_as_success(self, post):
        post.side_effect = [completion(""), completion(None), completion("")]
        self.assertIsNone(llm._chat("test prompt", max_tokens=256))
        self.assertEqual(post.call_count, len(llm.MODEL_CANDIDATES))

    @patch("citeguard.llm._chat", return_value="NO\nThe abstract is unrelated.")
    def test_judge_support_parses_no(self, chat):
        result = llm.judge_support("claim", "abstract")
        self.assertIs(result["supported"], False)
        self.assertIn("unrelated", result["reason"])

    @patch("citeguard.llm._chat", return_value="**YES**\nEvidence is present.")
    def test_judge_support_accepts_markdown_prefix(self, chat):
        result = llm.judge_support("claim", "abstract")
        self.assertIs(result["supported"], True)


if __name__ == "__main__":
    unittest.main()
