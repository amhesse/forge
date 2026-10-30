"""A bubblewrap sandbox for agents that run shell commands (design step 4).

What the agent can see:
- the item's worktree, read-write, as its working directory;
- /usr and /etc read-only (programs, certificates, DNS);
- a throwaway home directory, read-write and deleted afterwards;
- read-only extras the agent's own program needs (its binary, a provider
  package), plus the targets of the .venv / node_modules links that
  worktree.materialize() made, so the project's own tools still run.

What it can't see: the rest of $HOME (keys, other projects, the user's
credentials), the project's own checkout and .git, or other processes.

Network is shared, not filtered: the agent has to reach the model server on
localhost, and bubblewrap can't allow one endpoint and block the rest. That
is a known gap, stated rather than hidden. The filesystem is the boundary.
"""
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .. import worktree as wt

BWRAP = "bwrap"
# Top-level directories that are symlinks into /usr on merged-usr systems
# (Arch, Fedora); recreated as the same symlinks, or bound if they're real.
ROOT_LINKS = ("bin", "sbin", "lib", "lib64")


def available() -> bool:
    return shutil.which(BWRAP) is not None


def _materialized_targets(worktree: Path) -> list[Path]:
    """Real paths behind the .venv / venv / node_modules links in the worktree."""
    out = []
    for name in wt.MATERIALIZE_LINKS:
        link = worktree / name
        if link.is_symlink():
            out.append(link.resolve())
    return out


def bwrap_argv(worktree: Path, home: Path, ro: list[Path] = (),
               env: dict[str, str] | None = None) -> list[str]:
    worktree, home = worktree.resolve(), home.resolve()
    argv = [BWRAP, "--unshare-all", "--share-net", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for name in ROOT_LINKS:
        p = Path("/") / name
        if p.is_symlink():
            argv += ["--symlink", os.readlink(p), str(p)]
        elif p.is_dir():
            argv += ["--ro-bind", str(p), str(p)]
    for extra in [*ro, *_materialized_targets(worktree)]:
        if extra.exists():
            argv += ["--ro-bind", str(extra), str(extra)]
    argv += ["--bind", str(home), str(home), "--bind", str(worktree), str(worktree),
             "--chdir", str(worktree), "--clearenv"]
    base_env = {"HOME": str(home), "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": os.environ.get("LANG", "C.UTF-8"), "TERM": "dumb",
                "XDG_CONFIG_HOME": str(home / ".config"), "XDG_DATA_HOME": str(home / ".local/share"),
                "XDG_CACHE_HOME": str(home / ".cache"), "XDG_STATE_HOME": str(home / ".local/state")}
    for k, v in {**base_env, **(env or {})}.items():
        argv += ["--setenv", k, v]
    return argv


def run(cmd: list[str], worktree: Path, ro: list[Path] = (), env: dict[str, str] | None = None,
        home_setup=None, timeout: int = 1800, done_when=None, grace: float = 10.0
        ) -> subprocess.CompletedProcess:
    """Runs `cmd` inside the sandbox. `home_setup(home)` can populate the
    throwaway home first (a config file, a cache copy). Raises
    FileNotFoundError if bwrap isn't installed: an agent that runs shell
    commands never runs unsandboxed (see agents.get_agent), and
    subprocess.TimeoutExpired after `timeout` seconds.

    `done_when(line)` says, from one line of the agent's stdout, that it has
    given its final answer. From then on the sandbox gets `grace` seconds to
    exit before it's stopped, so a process the agent left running (a server,
    a watcher) can't hold finished work until the timeout. Stopping bwrap
    kills everything in the sandbox (--die-with-parent, and its PID
    namespace)."""
    if not available():
        raise FileNotFoundError("bwrap (bubblewrap) is not installed; it sandboxes agents that run commands")
    with tempfile.TemporaryDirectory(prefix="forge-sandbox-home-") as tmp:
        home = Path(tmp)
        if home_setup:
            home_setup(home)
        argv = bwrap_argv(worktree, home, list(ro), env) + list(cmd)
        # stdin is empty, never inherited: `opencode run` reads stdin when it
        # isn't a terminal, and forge started from a pipe that stays open
        # (a scheduler, a background job, a test runner) left it waiting on
        # that pipe forever. That was the hang seen on 2026-10-29.
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        out, err, done_at = [], [], []

        def read_out():
            for line in proc.stdout:
                out.append(line)
                if done_when and not done_at and done_when(line):
                    done_at.append(time.monotonic())

        readers = [threading.Thread(target=read_out, daemon=True),
                   threading.Thread(target=lambda: err.extend(proc.stderr), daemon=True)]
        for t in readers:
            t.start()
        deadline, lingered = time.monotonic() + timeout, False
        while proc.poll() is None:
            now = time.monotonic()
            if done_at and now - done_at[0] > grace:
                lingered = True
                _stop(proc)
                break
            if now > deadline:
                _stop(proc)
                for t in readers:
                    t.join(timeout=5)
                raise subprocess.TimeoutExpired(argv, timeout, "".join(out), "".join(err))
            time.sleep(0.2)
        for t in readers:
            t.join(timeout=5)
        returncode = 0 if lingered else proc.returncode
        if lingered:
            err.append(f"\n[forge] the agent finished but the sandbox hadn't exited {grace:.0f}s later; stopped it\n")
        return subprocess.CompletedProcess(argv, returncode, "".join(out), "".join(err))


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
