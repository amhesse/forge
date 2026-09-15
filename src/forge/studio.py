#!/usr/bin/env python3
"""
studio.py - Autonomous Software Studio & Real-Time Telemetry for Forge.

Industrial Monochrome Aesthetic with Signal Red Accents (Nothing OS / Technical instrument design).
Binds to 127.0.0.1:8888 by default.

Features:
  1. Live Execution Telemetry:
     - Pipeline stages (Idle -> Planning -> Worktree -> Coding -> Validating -> Merged/Parked)
     - Live token metrics (prompt/completion tokens, tokens/sec, peak rate, duration)
     - Hardware & Ollama telemetry (RTX GPU VRAM/temp/power, active Ollama models)
     - Live terminal console with log streaming via SSE
     - Real-time Todo Queue preview on Dashboard
  2. Interactive Run Controls:
     - Header control toolbar with model, backend, item limit selection and Start/Stop toggle
  3. Comprehensive TODO List & Spec Architect:
     - Full view of TODO.md / todo.md
     - Interactive Checklist view with one-click status toggling ([ ] <-> [x])
     - Raw Markdown Editor with Ctrl+S instant save and syntax line helpers
     - Side-by-Side Split view (Raw editor + Live parsed tasks & deterministic rule linter)
     - AI Goal Decomposer powered by local Ollama reasoning models
  4. Visual Diff Reviewer & Adversarial Critic:
     - High-contrast unified diff viewer with red deletions and green additions
     - AI-powered Adversarial Code Critic for automated risk assessment before merging
     - One-click Merge Anyway and Discard actions

Stdlib only: zero external dependencies, no build step. The page itself
lives in studio_static/ (index.html, app.css, app.js) as plain files; this
module fills index.html's `{{dotted.path}}` tokens and `__INITIAL_STATE__`
at request time and serves the rest from /static/.

Actions that would race a running loop (merge, discard, whole-checklist
save, requeue, project switch) are refused with 409 until it stops.
"""

import argparse
import contextlib
import datetime
import hashlib
import html
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config
from . import review_server
from . import worktree as wt

DEFAULT_PORT = 8888
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"


def ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host.rstrip("/")

# config.log() lines look like this; they already reach the UI through the
# log listener, so _LogTee must not forward them a second time.
_LOG_LINE_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


class _LogTee(io.TextIOBase):
    """stdout replacement for the runner thread: everything still goes to
    the real terminal, and bare print() lines (the ones config.log() didn't
    already deliver) are forwarded to the Studio log as whole lines."""

    def __init__(self, on_line):
        self._on_line = on_line
        self._buf = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        sys.__stdout__.write(s)
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._forward(line)
        return len(s)

    def flush(self) -> None:
        sys.__stdout__.flush()
        with self._lock:
            rest, self._buf = self._buf, ""
        self._forward(rest)

    def _forward(self, line: str) -> None:
        if line.strip() and not _LOG_LINE_RE.match(line):
            self._on_line(line)


def resolve_engine(value: str | None) -> tuple[str, str | None]:
    """Maps a model picker value to (engine, model).

    Picker values are `claude`, `claude:<model>`, `gemini`, `gemini:<model>`
    or a bare Ollama model name. A bare `claude`/`gemini` means "that CLI's
    own default model" - model is None, and no --model flag is passed."""
    value = (value or "").strip()
    for engine in ("claude", "gemini"):
        if value == engine:
            return engine, None
        if value.startswith(engine + ":"):
            return engine, value.split(":", 1)[1] or None
    return "ollama", value or None


def build_runner_argv(project_dir: Path, options: dict) -> list[str]:
    """Translates the Studio run controls into `forge run` arguments."""
    argv = ["--project-dir", str(project_dir),
            "--todo-file", find_todo_path(project_dir).name]

    backend = options.get("backend") or "lite"
    engine, model = resolve_engine(options.get("model"))
    if engine == "claude":
        backend = "claude"
        if model:
            argv += ["--models", model]
    elif engine == "gemini":
        # Gemini as the doer goes through aider, which names models
        # provider/model rather than provider:model.
        backend = "aider"
        argv += ["--models", f"gemini/{model}" if model else "gemini"]
    elif model:
        argv += ["--models", model]
    argv += ["--backend", backend]

    try:
        max_items = int(options.get("max_items") or 0)
    except (TypeError, ValueError):
        max_items = 0
    if max_items > 0:
        argv += ["--max-items", str(max_items)]

    if options.get("review"):
        engine, model = resolve_engine(options["review"])
        if engine == "claude":
            argv += ["--review", "claude"] + (["--review-model", model] if model else [])
        elif engine == "gemini":
            argv += ["--review", "agy"]
        else:
            argv += ["--review", "ollama", "--review-model", model]

    if options.get("fallback_model"):
        engine, model = resolve_engine(options["fallback_model"])
        if engine == "claude":
            argv += ["--fallback-backend", "claude"] + (["--fallback-model", model] if model else [])
        elif engine == "gemini":
            argv += ["--fallback-backend", "aider",
                     "--fallback-model", f"gemini/{model}" if model else "gemini"]
        else:
            argv += ["--fallback-model", model,
                     "--fallback-backend", options.get("fallback_backend") or backend]
    return argv


def describe_runner_options(options: dict) -> str:
    parts = [f"backend={options.get('backend') or 'lite'}",
             f"doer={options.get('model') or 'default'}"]
    if options.get("review"):
        parts.append(f"review={options['review']}")
    if options.get("fallback_model"):
        parts.append(f"escalate={options['fallback_model']}")
    if options.get("max_items"):
        parts.append(f"limit={options['max_items']}")
    return ", ".join(parts)


def run_llm(value: str | None, prompt: str, *, timeout: float = 600.0,
            num_predict: int = 2048, temperature: float = 0.2) -> str:
    """One-shot prompt to whichever engine a picker value names. Raises
    RuntimeError with the engine's own error text on failure."""
    engine, model = resolve_engine(value)
    if engine in ("claude", "gemini"):
        cmd = ["claude", "-p"] if engine == "claude" else ["agy", "--print"]
        if model and engine == "claude":
            cmd += ["--model", model]
        try:
            res = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise RuntimeError(f"`{cmd[0]}` CLI not found on PATH")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"`{cmd[0]}` timed out after {int(timeout)}s")
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout or f"exit {res.returncode}").strip())
        return res.stdout.strip()

    payload = json.dumps({
        "model": model or DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(f"{ollama_base_url()}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable: {e.reason}")


class StudioTelemetry:
    """Thread-safe telemetry state and event broker for Forge Studio."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.lock = threading.RLock()
        self.subscribers: list[queue.Queue] = []

        self.status = "idle"  # "idle", "running", "paused", "error"
        self.current_task = {
            "index": 0,
            "text": "",
            "stage": "idle",  # "idle", "worktree", "coding", "validating", "merged", "parked"
            "files": [],
            "started_at": None,
        }
        self.tokens = {
            "prompt": 0,
            "completion": 0,
            "total": 0,
            "tps": 0.0,
            "peak_tps": 0.0,
            "last_seconds": 0.0,
            "by_model": {},
        }
        self.gpu = {
            "name": "N/A",
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "util_pct": 0,
            "temp_c": 0,
            "power_w": 0.0,
        }
        self.ollama = {
            "loaded_model": "None",
            "context": 0,
            "until": "",
            "available_models": [],
        }
        self.recent_logs = deque(maxlen=500)
        self.recent_events = deque(maxlen=50)
        self.runner_thread: threading.Thread | None = None
        # Stop is only honoured between items, so a requested stop can sit
        # pending for as long as the current item takes - surfaced so the UI
        # can say "stopping after this item" instead of looking ignored.
        self.stop_pending = False

        # Wire into Forge's central logger, stage, and token emitters
        config.add_log_listener(self.on_log)
        config.add_stage_listener(self.set_stage)
        config.add_token_listener(self.update_tokens)

        # Start background hardware poller
        self._stop_poller = threading.Event()
        self._poller_thread = threading.Thread(target=self._hardware_poll_loop, daemon=True)
        self._poller_thread.start()

    def set_project_dir(self, project_dir: Path) -> None:
        with self.lock:
            self.project_dir = project_dir.resolve()
            self.broadcast("project_change", {"project_dir": str(self.project_dir)})

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=1000)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event_type: str, data: dict | str) -> None:
        # Serialized here, under the lock, rather than later on the SSE
        # thread: callers pass live dicts (tokens, gpu) that another thread
        # may be mutating, and json.dumps over a dict mid-mutation raises
        # "dictionary changed size during iteration".
        with self.lock:
            payload = {"event": event_type, "data_json": json.dumps(data)}
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def on_log(self, line: str) -> None:
        entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "msg": line,
        }
        with self.lock:
            self.recent_logs.append(entry)
        self.broadcast("log", entry)

    def set_stage(self, index: int, text: str, stage: str, files: list[str] | None = None) -> None:
        # UTC with milliseconds: parses identically in every browser, unlike
        # a naive local timestamp with microseconds.
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
        with self.lock:
            if stage in ("worktree", "coding", "validating"):
                self.status = "running"
            elif stage == "idle":
                self.status = "idle"
            # "worktree" is the first stage of every item, so the clock
            # restarts per item instead of counting from the run's first one.
            same_item = self.current_task.get("index") == index and stage != "worktree"
            started_at = self.current_task.get("started_at") if same_item else None
            self.current_task = {
                "index": index,
                "text": text,
                "stage": stage,
                "files": files or [],
                "started_at": started_at or now,
            }
            ev = {
                "index": index,
                "stage": stage,
                "text": text[:120],
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            }
            self.recent_events.append(ev)
        self.broadcast("stage", self.current_task)

    def update_tokens(self, prompt_tokens: int, completion_tokens: int, seconds: float = 0.0, model: str | None = None) -> None:
        with self.lock:
            self.tokens["prompt"] += prompt_tokens
            self.tokens["completion"] += completion_tokens
            self.tokens["total"] = self.tokens["prompt"] + self.tokens["completion"]
            tps = (completion_tokens / seconds) if seconds > 0 else 0.0
            self.tokens["tps"] = round(tps, 1)
            self.tokens["last_seconds"] = round(seconds, 2)
            if tps > self.tokens["peak_tps"]:
                self.tokens["peak_tps"] = round(tps, 1)

            if "by_model" not in self.tokens:
                self.tokens["by_model"] = {}
            m = model or self.ollama.get("loaded_model") or "primary"
            if m not in self.tokens["by_model"]:
                self.tokens["by_model"][m] = {"prompt": 0, "completion": 0, "total": 0}
            self.tokens["by_model"][m]["prompt"] += prompt_tokens
            self.tokens["by_model"][m]["completion"] += completion_tokens
            self.tokens["by_model"][m]["total"] += (prompt_tokens + completion_tokens)
        self.broadcast("tokens", self.tokens)

    def is_runner_active(self) -> bool:
        with self.lock:
            return self.runner_thread is not None and self.runner_thread.is_alive()

    def start_runner(self, options: dict) -> tuple[bool, str]:
        with self.lock:
            if self.is_runner_active():
                return False, "A runner task is already active"

            from . import runner

            argv = build_runner_argv(self.project_dir, options)
            config.reset_stop()
            self.stop_pending = False
            self.status = "running"
            self.broadcast("status", self._status_payload())
            summary = describe_runner_options(options)

            def _worker():
                # runner.main reports its own startup failures (no checklist,
                # not a git repo, dirty checkout) with print() + sys.exit,
                # which previously only ever reached the terminal Studio was
                # launched from - the UI just said "exited (1)".
                out = _LogTee(self.on_log)
                try:
                    self.on_log(f"Starting Forge loop: {summary}")
                    with contextlib.redirect_stdout(out):
                        code = runner.main(argv)
                    self.on_log(f"Forge loop finished (exit code {code}).")
                except SystemExit as se:
                    out.flush()
                    self.on_log(f"Forge runner exited ({se.code}).")
                except Exception as ex:
                    out.flush()
                    self.on_log(f"Forge runner error: {ex}")
                finally:
                    out.flush()
                    with self.lock:
                        self.status = "idle"
                        self.stop_pending = False
                        self.set_stage(0, "", "idle", [])
                        self.broadcast("status", self._status_payload(running=False))
                        self.broadcast("todo_updated", get_todo_details(self.project_dir))

            self.runner_thread = threading.Thread(target=_worker, daemon=True)
            self.runner_thread.start()
            return True, "Runner started"

    def stop_runner(self) -> tuple[bool, str]:
        if not self.is_runner_active():
            return True, "No active runner task"
        config.request_stop()
        with self.lock:
            self.stop_pending = True
            self.broadcast("status", self._status_payload())
        self.on_log("Stop requested - the loop will stop after the current item finishes.")
        return True, "Stopping after the current item"

    def _status_payload(self, running: bool | None = None) -> dict:
        with self.lock:
            is_running = self.is_runner_active() if running is None else running
            status = self.status
            if is_running and self.stop_pending:
                status = "stopping"
            return {"status": status, "is_running": is_running, "stop_pending": self.stop_pending}

    def get_state(self) -> dict:
        with self.lock:
            return json.loads(json.dumps({
                "project_dir": str(self.project_dir),
                "project_name": self.project_dir.name,
                **self._status_payload(),
                "current_task": self.current_task,
                "tokens": self.tokens,
                "gpu": self.gpu,
                "ollama": self.ollama,
                "recent_logs": list(self.recent_logs),
                "recent_events": list(self.recent_events),
            }))

    def _hardware_poll_loop(self) -> None:
        # Polls once up front so the first page load has real values, then
        # only while a browser is actually connected: nvidia-smi, `ollama ps`
        # and an HTTP call every 2s are pointless with nobody watching.
        first = True
        while not self._stop_poller.is_set():
            with self.lock:
                watched = bool(self.subscribers)
            if first or watched:
                self._poll_gpu()
                self._poll_ollama()
                first = False
            self._stop_poller.wait(2.0)

    def _poll_gpu(self) -> None:
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
                if len(parts) >= 7:
                    with self.lock:
                        self.gpu = {
                            "name": parts[0],
                            "total_mb": int(float(parts[1])),
                            "used_mb": int(float(parts[2])),
                            "free_mb": int(float(parts[3])),
                            "util_pct": int(float(parts[4])),
                            "temp_c": int(float(parts[5])),
                            "power_w": round(float(parts[6]), 1),
                        }
                    self.broadcast("gpu", self.gpu)
        except Exception:
            pass

    def _poll_ollama(self) -> None:
        loaded = "None"
        ctx = 0
        until = ""
        try:
            ps_res = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=2.0)
            if ps_res.returncode == 0:
                lines = [l for l in ps_res.stdout.splitlines() if l.strip()]
                if len(lines) > 1:
                    parts = lines[1].split()
                    if parts:
                        loaded = parts[0]
                    for idx, part in enumerate(parts):
                        if part.isdigit() and int(part) in (2048, 4096, 8192, 16384, 32768, 65536):
                            ctx = int(part)
        except Exception:
            pass

        avail = []
        try:
            req = urllib.request.Request(f"{ollama_base_url()}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                avail = [m["name"] for m in data.get("models", [])]
        except Exception:
            pass

        with self.lock:
            self.ollama = {
                "loaded_model": loaded,
                "context": ctx,
                "until": until,
                "available_models": avail,
            }
        self.broadcast("ollama", self.ollama)


def get_available_projects(current_dir: Path) -> list[str]:
    projects = set()
    if current_dir.is_dir():
        projects.add(str(current_dir.resolve()))
    home = Path.home()
    p_dir = home / "projects"
    if p_dir.is_dir():
        for item in p_dir.iterdir():
            if item.is_dir():
                if (item / ".git").is_dir():
                    projects.add(str(item.resolve()))
                try:
                    for sub in item.iterdir():
                        if sub.is_dir() and (sub / ".git").is_dir():
                            projects.add(str(sub.resolve()))
                except (PermissionError, OSError):
                    pass
    return sorted(list(projects))


def find_todo_path(project_dir: Path) -> Path:
    for name in ("TODO.md", "todo.md"):
        p = project_dir / name
        if p.is_file():
            return p
    return project_dir / "TODO.md"


def todo_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def get_todo_details(project_dir: Path) -> dict:
    """Returns structured todo file details, raw content, and parsed items.

    `hash` identifies the exact content the client was shown, so a save
    built from a stale copy (the runner has since written a status mark)
    can be refused instead of silently reverting that mark."""
    from . import todo, validator

    todo_p = find_todo_path(project_dir)
    content = todo_p.read_text(encoding="utf-8") if todo_p.is_file() else ""
    items = []
    if content:
        for idx, it in enumerate(todo.parse_todo_lines(content.splitlines()), 1):
            items.append({
                "index": idx,
                "line": it.line_index + 1,
                "status": it.status,
                "first_line": it.text.split("\n", 1)[0],
                "text": it.text,
                "files": validator.expected_files(it.text),
            })
    return {
        "filename": todo_p.name,
        "filepath": str(todo_p),
        "content": content,
        "hash": todo_hash(content),
        "exists": todo_p.is_file(),
        "items": items,
    }


def update_todo_item_status(project_dir: Path, line: int, new_status: str) -> bool:
    """Sets the status marker of the item whose checkbox is on `line`
    (1-indexed, as get_todo_details reports it)."""
    from . import todo

    if new_status not in (todo.STATUS_OPEN, todo.STATUS_DONE, todo.STATUS_BLOCKED,
                          todo.STATUS_NEEDS_REVIEW):
        raise ValueError(f"Invalid status {new_status!r}")
    todo_p = find_todo_path(project_dir)
    if not todo_p.is_file():
        return False
    _, items = todo.parse_todo(todo_p)
    for it in items:
        if it.line_index + 1 == line:
            todo.update_item_status(todo_p, it, new_status)
            return True
    return False


def requeue_stuck_items(project_dir: Path, lines: list[int] | None = None) -> int:
    """Flips `[!]` (blocked) and `[?]` (needs-review) items back to `[ ]`
    (open) so a fresh run - typically with a different model or backend -
    picks them up again. With `lines`, only the items on those (1-indexed)
    checkbox lines. Returns the number of items changed."""
    from . import todo

    todo_p = find_todo_path(project_dir)
    if not todo_p.is_file():
        return 0
    raw_lines, items = todo.parse_todo(todo_p)
    changed = 0
    for it in items:
        if lines is not None and it.line_index + 1 not in lines:
            continue
        if it.status in (todo.STATUS_BLOCKED, todo.STATUS_NEEDS_REVIEW):
            it.status = todo.STATUS_OPEN
            raw_lines[it.line_index] = it.render()
            changed += 1
    if changed:
        todo_p.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return changed


def mark_record_item_done(project_dir: Path, record: dict | None) -> bool:
    """After a manual merge, marks the checklist item that produced the
    branch `[x]`. Without this a merged item stayed `[?]`, and "Requeue
    Stuck" would then send already-merged work through the model again.

    Matched on full item text first (what the runner recorded), then on
    the first line alone in case continuation lines were edited since."""
    from . import todo

    item_text = (record or {}).get("item") or ""
    todo_p = find_todo_path(project_dir)
    if not item_text or not todo_p.is_file():
        return False
    _, items = todo.parse_todo(todo_p)
    first = item_text.split("\n", 1)[0].strip()
    match = next((it for it in items if it.text == item_text), None) or next(
        (it for it in items if it.text.split("\n", 1)[0].strip() == first), None)
    if match is None or match.status == todo.STATUS_DONE:
        return False
    todo.update_item_status(todo_p, match, todo.STATUS_DONE)
    return True


def lint_todo_content(content: str, project_dir: Path) -> dict:
    from . import spec_compiler, todo

    items = todo.parse_todo_lines(content.splitlines())
    results = []
    has_fatal = False
    for item in items:
        item_issues = []
        for msg, fatal in spec_compiler._validate_item_typed(item, project_dir):
            has_fatal = has_fatal or fatal
            item_issues.append({"message": msg, "fatal": fatal})
        results.append({
            "line": item.line_index + 1,
            "first_line": item.text.split("\n")[0][:100],
            "status": item.status,
            "issues": item_issues,
        })
    return {"valid": not has_fatal, "total_items": len(items), "results": results}


def draft_spec_goal(goal: str, project_dir: Path, model: str | None = None) -> dict:
    """Drafts checklist items for `goal`. Returns the draft plus its lint
    result - it is never written to the checklist here; the UI shows it
    for review first."""
    from . import spec_compiler

    todo_path = find_todo_path(project_dir)
    context = spec_compiler.gather_context(project_dir, todo_path, goal)
    prompt = spec_compiler.build_prompt(goal, context)
    engine, model_name = resolve_engine(model)
    try:
        if engine == "ollama":
            raw_draft = spec_compiler.call_ollama(
                model=model_name or DEFAULT_OLLAMA_MODEL,
                prompt=prompt,
                url=spec_compiler.DEFAULT_OLLAMA_URL,
                num_ctx=spec_compiler.DEFAULT_NUM_CTX,
                num_predict=spec_compiler.DEFAULT_NUM_PREDICT,
                timeout=spec_compiler.DEFAULT_TIMEOUT,
            )
        else:
            raw_draft = run_llm(model, prompt)
    except Exception as ex:
        return {"ok": False, "error": f"Failed to draft spec: {ex}"}
    draft = spec_compiler.strip_outer_fence(raw_draft).strip()
    return {"ok": True, "draft": draft, "model": model or DEFAULT_OLLAMA_MODEL,
            "lint": lint_todo_content(draft, project_dir)}


_VERDICT_RE = re.compile(r"\[\s*(APPROVE|CAUTION|REJECT)\b")


def critique_diff(branch: str, project_dir: Path, model: str | None = None) -> dict:
    rec = review_server.find_record_for_branch(project_dir, branch) or {}
    diff_text, diffstat = review_server.branch_diff(project_dir, branch, rec.get("base"))
    if not diff_text.strip():
        return {"ok": False, "error": "Diff is empty or branch has no changes"}

    prompt = f"""You are the Adversarial Code Reviewer and Critic for Forge, an autonomous software development studio.
Review the following git diff for this task:
TASK:
{rec.get("item", "Branch changes")}

PREVIOUS OUTCOME:
Status: {rec.get("status", "unknown")}
Reason: {rec.get("reason", "None specified")}

DIFFSTAT:
{diffstat}

UNIFIED DIFF:
{diff_text}

Please provide a sharp, structured code review in Markdown format:
### 1. Verdict
Must be one of:
- [APPROVE - SAFE TO MERGE]
- [CAUTION - MINOR RISKS]
- [REJECT - CRITICAL ISSUES]

### 2. Analysis of Changes
Briefly explain what the diff modifies.

### 3. Bugs, Flaws & Missing Coverage
Highlight any syntax issues, edge case handling problems, or missed tests.

### 4. Recommendation
Give concrete next steps (e.g. merge directly, or instructions to fix).
"""
    try:
        critique = run_llm(model, prompt, timeout=300.0)
    except Exception as e:
        return {"ok": False, "error": f"Critique failed: {e}"}
    # The template itself lists all three options, so a model that echoes
    # it would match all of them; the first bracketed verdict is the one
    # it chose far more often than not. Anything else is CAUTION.
    m = _VERDICT_RE.search(critique)
    verdict = m.group(1) if m else "CAUTION"
    return {"ok": True, "verdict": verdict, "critique": critique,
            "model": model or DEFAULT_OLLAMA_MODEL}


def index_run_records(project_dir: Path) -> dict[str, dict]:
    """branch -> newest run record, from one pass over every run directory.
    review_server.find_record_for_branch rescans everything per branch,
    which with many parked items made the branch list N x (runs x items)
    JSON parses - and the list now refreshes live on every item outcome."""
    root = wt.runs_root(project_dir)
    records: dict[str, dict] = {}
    if not root.is_dir():
        return records
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        for f in sorted(run_dir.glob("item-*.json"), reverse=True):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("branch"):
                records.setdefault(rec["branch"], rec)
    return records


def list_branch_summaries(project_dir: Path) -> list[dict]:
    """Parked branches without their diffs (fetched per branch on demand
    from /api/diff) - one `git diff --shortstat` each, nothing heavier."""
    records = index_run_records(project_dir)
    out = []
    for branch in review_server.list_pending_branches(project_dir):
        rec = records.get(branch) or {}
        base = rec.get("base")
        if not base:
            base = review_server.git(["merge-base", "HEAD", branch], project_dir).stdout.strip()
        stat = ""
        if base:
            stat = review_server.git(["diff", "--shortstat", f"{base}..{branch}"],
                                     project_dir).stdout.strip()
        out.append({
            "branch": branch,
            "status": rec.get("status", "unknown"),
            "reason": rec.get("reason", ""),
            "item": rec.get("item", ""),
            "item_title": (rec.get("item") or "").split("\n", 1)[0],
            "diffstat": stat,
        })
    return out


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.payload = {"ok": False, "error": message, **extra}


def _require_branch(branch) -> str:
    if not isinstance(branch, str) or not branch.startswith(review_server.BRANCH_PREFIX):
        raise ApiError(400, "Invalid branch")
    return branch


def _require_idle(telemetry: "StudioTelemetry", action: str) -> None:
    """Refuses actions that would race the runner: merges move HEAD under
    a loop that fast-forwards into it, and whole-file checklist writes can
    revert status marks it just wrote."""
    if telemetry.is_runner_active():
        raise ApiError(409, f"Stop the runner before you {action}.")


def merge_branch(project_dir: Path, branch: str) -> dict:
    todo_name = find_todo_path(project_dir).name
    clean, dirty = wt.is_clean(project_dir, exempt=(todo_name, "aider_loop.log"))
    if not clean:
        raise ApiError(409, "Working tree has uncommitted changes; commit or stash before merging.",
                       details=dirty)
    rec = index_run_records(project_dir).get(branch)
    res = review_server.git(["merge", "--no-ff", "-m", f"Merge {branch} (Forge Studio)", branch],
                            project_dir)
    if res.returncode != 0:
        review_server.git(["merge", "--abort"], project_dir)
        raise ApiError(409, "Merge failed - aborted, checkout unchanged.",
                       details=(res.stderr or res.stdout).strip())
    review_server.git(["branch", "-d", branch], project_dir)
    marked = mark_record_item_done(project_dir, rec)
    return {"ok": True, "branch": branch, "item_marked_done": marked}


def discard_branch(project_dir: Path, branch: str) -> dict:
    # A worktree still holding the branch (a killed run) makes -D refuse.
    wt.prune_stale(project_dir, log=lambda m: None)
    res = review_server.git(["branch", "-D", branch], project_dir)
    if res.returncode != 0:
        raise ApiError(409, "Could not delete branch.", details=(res.stderr or res.stdout).strip())
    return {"ok": True, "branch": branch}


MAX_BODY_BYTES = 5 * 1024 * 1024


def make_studio_handler(telemetry: StudioTelemetry):
    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress noisy standard HTTP request logging

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, handler):
            try:
                handler()
            except ApiError as e:
                self._send_json(e.payload, status=e.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                telemetry.on_log(f"Studio API error on {self.command} {self.path}: {e!r}")
                try:
                    self._send_json({"ok": False, "error": f"Internal error: {e}"}, status=500)
                except Exception:
                    pass

        def _read_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError(400, "Invalid Content-Length")
            if length > MAX_BODY_BYTES:
                raise ApiError(413, "Request body too large")
            body = self.rfile.read(length) if length > 0 else b""
            if not body:
                return {}
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise ApiError(400, f"Malformed JSON: {e}")
            if not isinstance(data, dict):
                raise ApiError(400, "JSON body must be an object")
            return data

        def do_GET(self):
            self._dispatch(self._get)

        def do_POST(self):
            self._dispatch(self._post)

        def _get(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            project = telemetry.project_dir

            if path in ("/", "/index.html"):
                self._send_bytes(render_studio_html(telemetry).encode("utf-8"),
                                 "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                asset = read_static_asset(path[len("/static/"):])
                if asset is None:
                    raise ApiError(404, "Not found")
                self._send_bytes(*asset)
            elif path == "/api/status":
                self._send_json(telemetry.get_state())
            elif path == "/api/projects":
                self._send_json({"projects": get_available_projects(project)})
            elif path == "/api/branches":
                self._send_json({"branches": list_branch_summaries(project)})
            elif path == "/api/diff":
                branch = _require_branch(query.get("branch", [""])[0])
                rec = index_run_records(project).get(branch) or {}
                diff_text, diffstat = review_server.branch_diff(project, branch, rec.get("base"))
                self._send_json({
                    "branch": branch,
                    "status": rec.get("status", "unknown"),
                    "reason": rec.get("reason", ""),
                    "item": rec.get("item", ""),
                    "diff": diff_text,
                    "diffstat": diffstat,
                })
            elif path == "/api/todo":
                self._send_json(get_todo_details(project))
            elif path == "/api/events":
                self._stream_events()
            else:
                raise ApiError(404, "Not found")

        def _stream_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = telemetry.subscribe()
            try:
                init = json.dumps(telemetry.get_state())
                self.wfile.write(f"event: init\ndata: {init}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        item = q.get(timeout=15.0)
                        self.wfile.write(
                            f"event: {item['event']}\ndata: {item['data_json']}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                telemetry.unsubscribe(q)

        def _post(self):
            path = urllib.parse.urlparse(self.path).path
            data = self._read_json()
            project = telemetry.project_dir

            def todo_changed(message: str) -> dict:
                details = get_todo_details(project)
                telemetry.on_log(message)
                telemetry.broadcast("todo_updated", details)
                return details

            if path == "/api/project":
                _require_idle(telemetry, "switch projects")
                raw = data.get("project_dir")
                if not isinstance(raw, str) or not raw.strip():
                    raise ApiError(400, "project_dir is required")
                new_path = Path(raw.strip()).expanduser().resolve()
                if not new_path.is_dir():
                    raise ApiError(400, f"Not a directory: {new_path}")
                telemetry.set_project_dir(new_path)
                self._send_json({"ok": True, "project_dir": str(new_path)})

            elif path == "/api/run":
                ok, msg = telemetry.start_runner(data)
                self._send_json({"ok": ok, "message": msg, "error": None if ok else msg},
                                status=200 if ok else 409)

            elif path == "/api/stop":
                ok, msg = telemetry.stop_runner()
                self._send_json({"ok": ok, "message": msg})

            elif path == "/api/ollama/start":
                try:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
                except FileNotFoundError:
                    raise ApiError(500, "`ollama` not found on PATH")
                time.sleep(0.6)
                telemetry._poll_ollama()
                self._send_json({"ok": True, "message": "Ollama service started.",
                                 "models": telemetry.ollama.get("available_models", [])})

            elif path == "/api/todo":
                _require_idle(telemetry, "save the whole checklist")
                content = data.get("content")
                if not isinstance(content, str):
                    raise ApiError(400, "content must be a string")
                todo_p = find_todo_path(project)
                current = todo_p.read_text(encoding="utf-8") if todo_p.is_file() else ""
                base_hash = data.get("base_hash")
                if base_hash and not data.get("force") and base_hash != todo_hash(current):
                    raise ApiError(409, f"{todo_p.name} changed on disk since you loaded it.",
                                   conflict=True, details=get_todo_details(project))
                todo_p.write_text(content, encoding="utf-8")
                details = todo_changed(f"Saved {todo_p.name} ({len(content.splitlines())} lines)")
                self._send_json({"ok": True, "filename": todo_p.name, "details": details})

            elif path == "/api/todo/toggle":
                try:
                    line = int(data.get("line"))
                except (TypeError, ValueError):
                    raise ApiError(400, "line must be an integer")
                status = data.get("status", "x")
                try:
                    found = update_todo_item_status(project, line, status)
                except ValueError as e:
                    raise ApiError(400, str(e))
                if not found:
                    raise ApiError(404, "Item not found")
                details = todo_changed(f"Updated item at line {line} -> [{status}]")
                self._send_json({"ok": True, "details": details})

            elif path == "/api/todo/requeue-stuck":
                _require_idle(telemetry, "requeue items")
                lines = data.get("lines")
                if lines is not None:
                    if not isinstance(lines, list) or not all(isinstance(n, int) for n in lines):
                        raise ApiError(400, "lines must be a list of integers")
                count = requeue_stuck_items(project, lines)
                details = todo_changed(f"Requeued {count} stuck item(s) ([!]/[?] -> [ ]).")
                self._send_json({"ok": True, "count": count, "details": details})

            elif path == "/api/todo/add":
                task_text = data.get("task")
                if not isinstance(task_text, str) or not task_text.strip():
                    raise ApiError(400, "Task text cannot be empty")
                task_text = task_text.strip()
                todo_p = find_todo_path(project)
                old = todo_p.read_text(encoding="utf-8") if todo_p.is_file() else ""
                todo_p.write_text((old.rstrip() + "\n\n" + task_text + "\n") if old.strip()
                                  else task_text + "\n", encoding="utf-8")
                details = todo_changed(
                    f"Added task to {todo_p.name}: {task_text.splitlines()[0][:60]}")
                self._send_json({"ok": True, "details": details})

            elif path == "/api/lint-spec":
                content = data.get("content", "")
                if not isinstance(content, str):
                    raise ApiError(400, "content must be a string")
                self._send_json(lint_todo_content(content, project))

            elif path == "/api/draft-spec":
                goal = data.get("goal")
                if not isinstance(goal, str) or not goal.strip():
                    raise ApiError(400, "Goal cannot be empty")
                self._send_json(draft_spec_goal(goal.strip(), project, data.get("model")))

            elif path == "/api/critic":
                branch = _require_branch(data.get("branch"))
                self._send_json(critique_diff(branch, project, data.get("model")))

            elif path == "/api/merge":
                _require_idle(telemetry, "merge a branch")
                result = merge_branch(project, _require_branch(data.get("branch")))
                telemetry.on_log(f"Merged {result['branch']}"
                                 + (" and marked its item [x]." if result["item_marked_done"] else "."))
                telemetry.broadcast("branches_changed", {})
                telemetry.broadcast("todo_updated", get_todo_details(project))
                self._send_json(result)

            elif path == "/api/discard":
                _require_idle(telemetry, "discard a branch")
                result = discard_branch(project, _require_branch(data.get("branch")))
                telemetry.on_log(f"Discarded {result['branch']}.")
                telemetry.broadcast("branches_changed", {})
                self._send_json(result)

            else:
                raise ApiError(404, "Not found")

    return StudioHandler


class StudioServer:
    def __init__(self, project_dir: Path, port: int = DEFAULT_PORT):
        self.project_dir = project_dir.resolve()
        self.port = port
        self.telemetry = StudioTelemetry(self.project_dir)
        handler_cls = make_studio_handler(self.telemetry)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        self._thread = None

    def start(self, in_background: bool = True):
        if in_background:
            self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self._thread.start()
        else:
            self.httpd.serve_forever()

    def stop(self):
        self.telemetry._stop_poller.set()
        self.httpd.shutdown()
        self.httpd.server_close()


STATIC_DIR = Path(__file__).with_name("studio_static")
STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}
_TEMPLATE_TOKEN_RE = re.compile(r"\{\{([a-z_]+(?:\.[a-z_]+)*)\}\}")


def render_studio_html(telemetry: StudioTelemetry) -> str:
    """Renders index.html with the current state: `__INITIAL_STATE__`
    becomes the JSON state blob, and `{{dotted.path}}` tokens become
    HTML-escaped first-paint values (the JS overwrites them on load)."""
    state = telemetry.get_state()
    template = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def lookup(match: re.Match) -> str:
        value = state
        for key in match.group(1).split("."):
            value = value.get(key, "") if isinstance(value, dict) else ""
        return html.escape(str(value))

    page = _TEMPLATE_TOKEN_RE.sub(lookup, template)
    initial_json = json.dumps(state).replace("</", "<\\/")
    return page.replace("__INITIAL_STATE__", initial_json)


def read_static_asset(name: str) -> tuple[bytes, str] | None:
    """Returns (body, content type) for a file directly inside
    studio_static/, or None for anything else - no subdirectories, no
    traversal, only the known asset types."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    path = STATIC_DIR / name
    ctype = STATIC_TYPES.get(path.suffix)
    if ctype is None or not path.is_file():
        return None
    return path.read_bytes(), ctype


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forge Studio: real-time visual telemetry, spec architect, and workspace dashboard"
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        type=str,
        help="Project directory to manage (default: current dir)",
    )
    parser.add_argument(
        "--port", default=DEFAULT_PORT, type=int, help=f"HTTP port (default: {DEFAULT_PORT})"
    )
    parser.add_argument("--open", action="store_true", help="Automatically open browser")
    args = parser.parse_args(argv)

    project_dir = Path(args.project_dir).expanduser().resolve()
    print(f"Starting Forge Studio on http://127.0.0.1:{args.port}/ (managing {project_dir})")
    server = StudioServer(project_dir=project_dir, port=args.port)

    if args.open:
        try:
            subprocess.run(["xdg-open", f"http://127.0.0.1:{args.port}/"], check=False)
        except Exception:
            pass

    try:
        server.start(in_background=False)
    except KeyboardInterrupt:
        print("\nStopping Forge Studio...")
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
