import httpx
import pytest

from valta.ollama_client import OllamaClient, OllamaError, strip_thinking


class _Response:
    def __init__(self, status_code=200, message=None, text=""):
        self.status_code = status_code
        self._message = message or {}
        self.text = text

    def json(self):
        return {"message": self._message}


def _serve(monkeypatch, *responses):
    sent = []
    queue = list(responses)

    def fake_post(url, json, timeout):
        sent.append(json)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_chat_turns_thinking_off_and_returns_only_the_answer(monkeypatch):
    sent = _serve(monkeypatch, _Response(message={"content": "<think>\nhmm\n</think>\n\nTALK\nHi.", "thinking": ""}))
    assert OllamaClient(settings=None).chat("http://h", "qwen3.5:9b", [{"role": "user", "content": "hi"}], 8192) == "TALK\nHi."
    assert sent[0]["think"] is False
    assert sent[0]["options"] == {"num_ctx": 8192, "temperature": 0.4}


def test_a_reply_that_only_thought_is_a_failure(monkeypatch):
    _serve(monkeypatch, _Response(message={"content": "", "thinking": "Let me consider..."}))
    with pytest.raises(OllamaError, match="only thought"):
        OllamaClient(settings=None).chat("http://h", "m", [], 4096)


def test_an_ollama_that_refuses_the_think_field_is_asked_again_without_it(monkeypatch):
    sent = _serve(
        monkeypatch,
        _Response(400, text='{"error":"invalid field think"}'),
        _Response(message={"content": "42"}),
    )
    assert OllamaClient(settings=None).chat("http://h", "old", [], 4096) == "42"
    assert "think" in sent[0] and "think" not in sent[1]


def test_a_connection_failure_is_an_ollama_error(monkeypatch):
    _serve(monkeypatch, httpx.ConnectError("refused"))
    with pytest.raises(OllamaError, match="refused"):
        OllamaClient(settings=None).chat("http://h", "m", [], 4096)


def test_strip_thinking_leaves_ordinary_text_alone():
    assert strip_thinking("def f():\n    return '<think>'\n") == "def f():\n    return '<think>'\n"
    assert strip_thinking("  <think>a\nb</think>answer") == "answer"
