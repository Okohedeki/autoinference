"""API server part: a minimal OpenAI-compatible chat endpoint on the standard library.

One background thread owns the engine and runs its step loop, so concurrent
HTTP requests are batched together by the scheduler.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Engine
from .parts.renderer import IncrementalDetokenizer
from .types import SamplingParams, TokenEvent


class EngineLoop:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._pending: queue.Queue[tuple[list[int], SamplingParams, queue.Queue]] = queue.Queue()
        self._streams: dict[int, queue.Queue] = {}
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="engine-loop")
        self._thread.start()

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> queue.Queue:
        out: queue.Queue[TokenEvent | Exception] = queue.Queue()
        self._pending.put((prompt_ids, params, out))
        self._wake.set()
        return out

    def _run(self) -> None:
        while True:
            while not self._pending.empty():
                prompt_ids, params, out = self._pending.get()
                try:
                    rid = self.engine.add_request(prompt_ids, params)
                    self._streams[rid] = out
                except Exception as exc:  # reported to the waiting handler
                    out.put(exc)
            if not self.engine.has_work():
                self._wake.wait()
                self._wake.clear()
                continue
            events = self.engine.step()
            for ev in events.tokens:
                stream = self._streams.get(ev.request_id)
                if stream is None:
                    continue
                stream.put(ev)
                if ev.finished:
                    del self._streams[ev.request_id]
                    self.engine.scheduler.requests.pop(ev.request_id, None)


def _sampling_from(body: dict) -> SamplingParams:
    return SamplingParams(
        max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 256),
        temperature=float(body.get("temperature", 0.0)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0)),
        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
        seed=body.get("seed"),
    )


def make_handler(loop: EngineLoop, model_name: str):
    engine = loop.engine

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # quieter default logging
            pass

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt_ids = engine.renderer.apply_chat_template(body["messages"])
                params = _sampling_from(body)
            except Exception as exc:
                self._json(400, {"error": f"bad request: {exc}"})
                return
            out = loop.submit(prompt_ids, params)
            created = int(time.time())
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            detok = IncrementalDetokenizer(engine.renderer)
            if body.get("stream"):
                self._stream(out, detok, cid, created)
            else:
                self._complete(out, detok, cid, created, len(prompt_ids))

        def _events(self, out: queue.Queue):
            while True:
                ev = out.get()
                if isinstance(ev, Exception):
                    raise ev
                yield ev
                if ev.finished:
                    return

        def _complete(self, out, detok, cid, created, prompt_len) -> None:
            text, n, reason = "", 0, None
            try:
                for ev in self._events(out):
                    n += 1
                    reason = ev.finish_reason
                    if ev.token_id not in engine.eos_token_ids:
                        text += detok.add(ev.token_id)
            except Exception as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": n, "total_tokens": prompt_len + n},
            })

        def _stream(self, out, detok, cid, created) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(delta: dict, reason: str | None) -> None:
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            send({"role": "assistant"}, None)
            for ev in self._events(out):
                piece = "" if ev.token_id in engine.eos_token_ids else detok.add(ev.token_id)
                send({"content": piece} if piece else {}, ev.finish_reason if ev.finished else None)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000, model_name: str | None = None) -> None:
    if engine.renderer is None:
        raise ValueError("serving needs a model with tokenizer files")
    loop = EngineLoop(engine)
    name = model_name or engine.model_dir.name
    httpd = ThreadingHTTPServer((host, port), make_handler(loop, name))
    print(f"serving {name} on http://{host}:{port}/v1/chat/completions")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
