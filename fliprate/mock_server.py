"""MOCK OpenAI-compatible server for smoke tests ONLY. Its outputs are synthetic and are NOT results.

The fake 'model' answers deterministically (hash-based), and on rebuttals flips to the
pushed letter with probability rising by escalation level. Run:  python -m fliprate.mock_server --port 8765
"""
from __future__ import annotations
import argparse, hashlib, json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _h(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def fake_reply(messages: list[dict]) -> str:
    user = [m["content"] for m in messages if m["role"] == "user"]
    first = user[0]
    level = len(user) - 1
    base = "ABCD"[int(_h("a" + first) * 4)]
    if level == 0:
        return f"ANSWER: {base}\nCONFIDENCE: {60 + int(_h('c' + first) * 30)}"
    m = re.search(r"(?:answer is|believe the answer is|correct it to|to) ([ABCD])\b", user[level])
    push = m.group(1) if m else base
    if _h(f"flip{level}{first}") < 0.25 * level:
        return f"ANSWER: {push}\nCONFIDENCE: {max(10, 70 - 10 * level)}"
    return f"ANSWER: {base}\nCONFIDENCE: {max(10, 85 - 5 * level)}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"data": [{"id": "mock-model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        txt = fake_reply(body["messages"])
        self._send({"choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}]})


def start(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"MOCK server on http://127.0.0.1:{a.port}/v1 (synthetic outputs, not results)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
