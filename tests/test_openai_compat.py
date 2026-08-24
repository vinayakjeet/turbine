import json

import httpx
import pytest

from turbine.backends.base import CompletionError
from turbine.backends.openai_compat import OpenAICompatBackend
from turbine.clocks import RealClock


def sse_transport(payload_chunks, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, content=b"model not found")
        lines = [f"data: {json.dumps({'choices': [{'text': t}]})}\n\n" for t in payload_chunks]
        lines.append("data: [DONE]\n\n")
        body = "".join(lines).encode()
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def backend_for(chunks, status=200) -> OpenAICompatBackend:
    client = httpx.AsyncClient(transport=sse_transport(chunks, status), base_url="http://srv")
    return OpenAICompatBackend("http://srv", "test-model", client=client, clock=RealClock())


async def test_parses_sse_and_stops_at_done():
    out = []
    async for chunk in backend_for(["Hello", " world"]).stream("p", max_tokens=8, request_seed=1):
        out.append(chunk)
    assert "".join(out) == "Hello world"


async def test_http_error_becomes_completion_error():
    with pytest.raises(CompletionError, match="404"):
        async for _ in backend_for([], status=404).stream("p", max_tokens=8, request_seed=1):
            pass


async def test_connection_failure_is_counted_not_raised_past_the_boundary():
    class Exploding(httpx.AsyncClient):
        async def send(self, *a, **kw):
            raise httpx.ConnectError("nope")

    b = OpenAICompatBackend(
        "http://srv",
        "m",
        client=Exploding(base_url="http://srv"),
        clock=RealClock(),
    )
    with pytest.raises(CompletionError, match="connection failed"):
        async for _ in b.stream("p", max_tokens=8, request_seed=1):
            pass


async def test_seed_travels_in_the_payload():
    seen = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = 'data: {"choices": [{"text": "x"}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(capture), base_url="http://srv")
    b = OpenAICompatBackend("http://srv", "m", client=client, clock=RealClock())
    async for _ in b.stream("prompt text", max_tokens=5, request_seed=42):
        pass
    assert seen["seed"] == 42
    assert seen["max_tokens"] == 5
    assert seen["stream"] is True
