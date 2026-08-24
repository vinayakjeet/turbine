"""Client for any OpenAI-compatible completions endpoint.

That is the surface `vllm serve` exposes, which is the only integration
this repo needs: measurements run against a server process, never against
in-process model code, so the harness cannot influence what it measures.

Each non-empty delta counts as one token. Servers report authoritative
usage in the final chunk; the approximation is fine for TTFT and TPOT,
which are clock-derived anyway.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from turbine.backends.base import CompletionError
from turbine.clocks import Clock, RealClock


class OpenAICompatBackend:
    name = "openai-compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._model = model
        self._clock = clock or RealClock()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            # Read timeout is None on purpose: the per-request deadline lives
            # in the load generator's clock so it applies to mock and live
            # runs identically. A transport-level read timeout would measure
            # httpx, not the server.
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        )

    async def stream(
        self, prompt: str, *, max_tokens: int, request_seed: int
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "seed": request_seed,
        }
        request = self._client.build_request("POST", "/v1/completions", json=payload)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise CompletionError(f"connection failed: {exc!r}") from exc
        try:
            if response.status_code != 200:
                body = (await response.aread())[:200]
                raise CompletionError(f"HTTP {response.status_code}: {body!r}")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise CompletionError(f"unparseable SSE payload: {data[:120]!r}") from exc
                text = chunk.get("choices", [{}])[0].get("text")
                if text:
                    yield text
        except httpx.HTTPError as exc:
            raise CompletionError(f"stream failed: {exc!r}") from exc
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()
