from __future__ import annotations

import hashlib
import json
from pathlib import Path
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

    def chat(
        self,
        host: str,
        model: str,
        messages: list[dict[str, str]],
        num_ctx: int,
        temperature: float = 0.4,
    ) -> str:
        response = httpx.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": num_ctx, "temperature": temperature},
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

    def delete(self, host: str, name: str) -> None:
        try:
            response = httpx.request("DELETE", f"{host}/api/delete", json={"model": name}, timeout=60)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not delete {name} on {host}: {exc}") from exc
        if response.status_code >= 400 and response.status_code != 404:
            raise OllamaError(f"delete {name} failed ({response.status_code}): {response.text[:400]}")

    def create(self, host: str, name: str, modelfile: str) -> None:
        """Create a model from Modelfile text (FROM and PARAMETER lines).

        Current Ollama no longer takes a raw `modelfile` field. A local GGUF is uploaded as a blob
        and named under `files`; any other FROM is passed as `from`.
        """
        source, parameters = parse_modelfile(modelfile)
        if not source:
            raise OllamaError(f"create {name} failed: the Modelfile has no FROM line")
        body: dict[str, Any] = {"model": name, "stream": True}
        if parameters:
            body["parameters"] = parameters
        path = Path(source)
        if path.is_file():
            body["files"] = {path.name: self.push_blob(host, path)}
        else:
            body["from"] = source
        with httpx.stream(
            "POST",
            f"{host}/api/create",
            json=body,
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

    def push_blob(self, host: str, path: Path) -> str:
        """Upload a file to the server's blob store, unless it is already there. Returns its digest."""
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
        digest = f"sha256:{sha.hexdigest()}"
        url = f"{host}/api/blobs/{digest}"
        try:
            if httpx.head(url, timeout=30).status_code == 200:
                return digest
            print(f"  uploading {path.name} to {host}", flush=True)
            with path.open("rb") as handle:
                response = httpx.post(url, content=_chunks(handle), timeout=None)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not upload {path.name} to {host}: {exc}") from exc
        if response.status_code >= 400:
            raise OllamaError(f"upload {path.name} failed ({response.status_code}): {response.text[:400]}")
        return digest


def _chunks(handle, size: int = 1 << 20):
    for block in iter(lambda: handle.read(size), b""):
        yield block


def parse_modelfile(text: str) -> tuple[str, dict[str, Any]]:
    """The FROM source and PARAMETER values of a simple Modelfile. Numbers become numbers."""
    source = ""
    parameters: dict[str, Any] = {}
    for line in text.splitlines():
        keyword, _, rest = line.strip().partition(" ")
        rest = rest.strip()
        if keyword.upper() == "FROM":
            source = rest
        elif keyword.upper() == "PARAMETER":
            key, _, value = rest.partition(" ")
            parameters[key] = _number(value.strip())
    return source, parameters


def _number(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
