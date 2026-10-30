"""A scripted stand-in for an OpenAI-compatible model server (Ollama's /v1,
llama-server), so a real agent harness can be driven end to end without a
GPU. Each request to /v1/chat/completions gets the next scripted reply,
streamed as server-sent events the way the real servers stream them.

A reply is either {"text": "..."} or {"tool": "name", "args": {...}}; when
the script runs out, the last reply repeats. Only requests that offer tools
(the agent's own turns) use the script; others get a plain reply. Every
request body is kept in `requests` for the test to inspect.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAI:
    def __init__(self, script: list[dict], model: str = "fake-model"):
        self.script, self.model, self.requests = list(script), model, []
        self._lock = threading.Lock()
        self._n = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._json({"object": "list", "data": [{"id": server.model, "object": "model"}]})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with server._lock:
                    server.requests.append(body)
                    if not body.get("tools"):
                        # Side calls (OpenCode titles its session with one) don't
                        # use up the agent's script.
                        reply = {"text": "Session title"}
                    else:
                        reply = server.script[min(server._n, len(server.script) - 1)]
                        server._n += 1
                if body.get("stream"):
                    self._stream(reply)
                else:
                    self._json(server._completion(reply))

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, reply):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in server._chunks(reply):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()

    def _base(self):
        return {"id": f"chatcmpl-{self._n}", "object": "chat.completion.chunk", "created": 0, "model": self.model}

    def _chunks(self, reply):
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        if "tool" in reply:
            call = {"index": 0, "id": f"call_{self._n}", "type": "function",
                    "function": {"name": reply["tool"], "arguments": json.dumps(reply["args"])}}
            yield {**self._base(), "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [call]},
                                                "finish_reason": None}]}
            yield {**self._base(), "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": usage}
        else:
            yield {**self._base(), "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply["text"]},
                                                "finish_reason": None}]}
            yield {**self._base(), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}

    def _completion(self, reply):
        msg = {"role": "assistant", "content": reply.get("text", "")}
        if "tool" in reply:
            msg["tool_calls"] = [{"id": "call_0", "type": "function",
                                  "function": {"name": reply["tool"], "arguments": json.dumps(reply["args"])}}]
        return {"id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": self.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if "tool" in reply else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
