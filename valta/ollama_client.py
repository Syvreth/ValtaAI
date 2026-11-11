from __future__ import annotations

import json
from typing import Any

import httpx

from valta.config import Settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings, timeout: float = 600):
        self.settings = settings
        self.timeout = timeout

    def healthy(self, host: str) -> bool:
        try:
            response = httpx.get(f"{host}/api/version", timeout=3)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def version(self, host: str) -> str:
        response = httpx.get(f"{host}/api/version", timeout=5)
        response.raise_for_status()
        return response.json().get("version", "unknown")

    def tags(self, host: str) -> list[dict[str, Any]]:
        response = httpx.get(f"{host}/api/tags", timeout=10)
        response.raise_for_status()
        return list(response.json().get("models", []))

    def ps(self, host: str) -> list[dict[str, Any]]:
        response = httpx.get(f"{host}/api/ps", timeout=10)
        response.raise_for_status()
        return list(response.json().get("models", []))

    def chat(self, host: str, model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
        response = httpx.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": num_ctx, "temperature": 0.4},
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise OllamaError(f"{host} chat failed ({response.status_code}): {response.text[:400]}")
        content = response.json().get("message", {}).get("content", "")
        if not content:
            raise OllamaError(f"{host} returned an empty reply from {model}")
        return content

    def embed(self, host: str, model: str, text: str) -> list[float]:
        response = httpx.post(
            f"{host}/api/embed",
            json={"model": model, "input": text},
            timeout=120,
        )
        if response.status_code >= 400:
            raise OllamaError(f"{host} embed failed ({response.status_code}): {response.text[:400]}")
        payload = response.json()
        if payload.get("embeddings"):
            return list(payload["embeddings"][0])
        if payload.get("embedding"):
            return list(payload["embedding"])
        raise OllamaError("Ollama embed response had no vector")

    def pull(self, host: str, model: str) -> None:
        with httpx.stream(
            "POST",
            f"{host}/api/pull",
            json={"name": model, "stream": True},
            timeout=None,
        ) as response:
            if response.status_code >= 400:
                raise OllamaError(f"pull {model} failed: {response.read().decode()[:400]}")
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                status = payload.get("status", "")
                if payload.get("error"):
                    raise OllamaError(str(payload["error"]))
                total = payload.get("total")
                completed = payload.get("completed")
                if total and completed:
                    print(f"  {model}: {status} {completed}/{total}", flush=True)
                elif status:
                    print(f"  {model}: {status}", flush=True)

    def unload(self, host: str, model: str) -> None:
        try:
            httpx.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not unload {model} on {host}: {exc}") from exc

    def create(self, host: str, name: str, modelfile: str) -> None:
        with httpx.stream(
            "POST",
            f"{host}/api/create",
            json={"model": name, "modelfile": modelfile, "stream": True},
            timeout=None,
        ) as response:
            if response.status_code >= 400:
                raise OllamaError(f"create {name} failed: {response.read().decode()[:400]}")
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("error"):
                    raise OllamaError(str(payload["error"]))
                status = payload.get("status")
                if status:
                    print(f"  create {name}: {status}", flush=True)
