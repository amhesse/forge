#!/usr/bin/env python3
"""
review_server.py

A local page for looking at what the loop parked overnight and deciding
what to do with each one, instead of re-reading terminal scrollback or
running `git log -p` by hand for every branch the run summary named.

    python review_server.py --project-dir ~/projects/blockroad

Then open http://127.0.0.1:8765/. Ctrl+C to stop; nothing here runs
unattended or writes anything until you click a button.

The branch IS the item, not the JSON log of it: this reads
`git branch --list 'aider-loop/*'` as the list of pending work, and only
uses aider_loop's own run records (~/.cache/aider-loop/<project>/runs/)
for context - the item's text, why it was parked - when a record for that
branch still exists. A branch can outlive its run record (an old cache
directory cleaned up separately) or the reverse (a branch already merged
or deleted by hand), and the git branches are what's actually still
sitting there needing a decision; a JSON file is only ever commentary on
one.

Two actions, both real git operations on the project's actual checkout,
confirmed in the browser before they run:

  - Merge anyway: `git merge --no-ff <branch>`. Parked items were never
    applied to the checkout (that's the whole point of the worktree
    design - see aider_loop's README), so nothing here is undoing a
    revert; this is a human overriding "needs review" or "blocked" after
    actually reading the diff, the same as manually running the merge.
    A conflict is reported back, not silently resolved.
  - Discard: `git branch -D <branch>`. Only ever removes the branch - the
    project checkout was never touched by a parked item and isn't
    touched by discarding one either.

No auth, because there's nothing here that isn't already a `git` command
this user could run directly - this just saves the typing and shows the
diff alongside the decision. Binds to 127.0.0.1 only; it is not meant to
be reachable from another machine.

Stdlib only, matching aider_loop.py's own requirement - no Flask, no
Jinja, so a browser tab and a project directory are the only things this
needs.
"""

import argparse
import datetime
import html
import json
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import worktree as wt

DEFAULT_PORT = 8765
BRANCH_PREFIX = "aider-loop/"


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def list_pending_branches(project_dir: Path) -> list[str]:
    result = git(["branch", "--list", f"{BRANCH_PREFIX}*", "--format=%(refname:short)"], project_dir)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def find_record_for_branch(project_dir: Path, branch: str) -> dict | None:
    """Searches every run's item-*.json for one whose branch matches.
    Not indexed - a project accumulates at most a few hundred of these
    between cleanups, and this runs once per page load, not per request
    in a hot loop."""
    runs_root = wt.runs_root(project_dir)
    if not runs_root.is_dir():
        return None
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        for f in run_dir.glob("item-*.json"):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("branch") == branch:
                return rec
    return None


def branch_diff(project_dir: Path, branch: str, base: str | None) -> tuple[str, str]:
    """Returns (diff_text, diffstat_text). Diffs against the record's own
    base commit when known (the commit the item actually branched from),
    falling back to the merge-base with the current HEAD - a branch
    without a surviving record still deserves a diff, just computed the
    way `git` itself would pick a comparison point."""
    if not base:
        mb = git(["merge-base", "HEAD", branch], project_dir)
        base = mb.stdout.strip() or None
    if not base:
        return "(could not determine a base commit to diff against)", ""
    diff = git(["diff", f"{base}..{branch}"], project_dir)
    stat = git(["diff", "--stat", f"{base}..{branch}"], project_dir)
    return diff.stdout, stat.stdout


def recent_merged(project_dir: Path, limit: int = 8) -> list[str]:
    result = git(["log", f"-{limit}", "--oneline"], project_dir)
    return [line for line in result.stdout.splitlines() if line.strip()]


STATUS_LABEL = {"!": "blocked", "?": "needs review", None: "unknown (no run record)"}


def render_page(project_dir: Path, message: str | None = None) -> str:
    branches = list_pending_branches(project_dir)
    items = []
    for branch in branches:
        rec = find_record_for_branch(project_dir, branch)
        diff, stat = branch_diff(project_dir, branch, rec.get("base") if rec else None)
        items.append({"branch": branch, "record": rec, "diff": diff, "stat": stat})

    esc = html.escape
    rows = []
    for it in items:
        rec = it["record"] or {}
        status = STATUS_LABEL.get(rec.get("status"), esc(str(rec.get("status"))))
        item_text = esc(rec.get("item", "(no run record for this branch)"))
        reason = esc(rec.get("reason", "-"))
        branch = esc(it["branch"])
        tdd = rec.get("tdd")
        tdd_note = (f'<div class="tdd">TDD item &middot; test <code>{esc(tdd["test_file"])}</code> '
                    f'&middot; impl <code>{esc(tdd["impl_file"])}</code></div>') if tdd else ""
        val_out = rec.get("validation_output")
        val_block = (f'<details><summary>Validation output</summary><pre>{esc(val_out)}</pre></details>'
                    if val_out else "")
        rows.append(f"""
        <div class="item">
          <div class="item-head">
            <span class="status status-{esc(str(rec.get('status', '?')))}">{status}</span>
            <code class="branch">{branch}</code>
          </div>
          <div class="text">{item_text}</div>
          {tdd_note}
          <div class="reason"><b>Why it's here:</b> {reason}</div>
          {val_block}
          <details {"open" if it["diff"] else ""}>
            <summary>Diff ({esc(" · ".join(l.strip() for l in it["stat"].strip().splitlines()) or "no diff")})</summary>
            <pre class="diff">{esc(it["diff"]) or "(empty diff)"}</pre>
          </details>
          <form method="post" action="/action" class="actions">
            <input type="hidden" name="branch" value="{branch}">
            <button name="do" value="merge" class="merge"
                    onclick="return confirm('Merge {branch} into the current branch with --no-ff?');">
              Merge anyway
            </button>
            <button name="do" value="discard" class="discard"
                    onclick="return confirm('Permanently delete branch {branch}? This cannot be undone.');">
              Discard
            </button>
          </form>
        </div>""")

    body = "\n".join(rows) if rows else '<p class="empty">Nothing parked. Every item either passed or the queue is empty.</p>'
    recent = "\n".join(f"<li><code>{esc(l)}</code></li>" for l in recent_merged(project_dir))
    banner = f'<div class="banner">{esc(message)}</div>' if message else ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>aider-loop review — {esc(project_dir.name)}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;
         background: #f7f7f8; color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; }}
  .banner {{ background: #eaffea; border: 1px solid #9c9; padding: 0.5rem 1rem; border-radius: 6px; margin-bottom: 1rem; }}
  .item {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  .item-head {{ display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.4rem; }}
  .status {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
             padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; }}
  .status-\\! {{ background: #ffe0e0; color: #a00; }}
  .status-\\? {{ background: #fff3cd; color: #8a6100; }}
  .branch {{ color: #666; font-size: 0.85rem; }}
  .text {{ white-space: pre-wrap; margin-bottom: 0.5rem; }}
  .tdd {{ font-size: 0.85rem; color: #555; margin-bottom: 0.5rem; }}
  .reason {{ font-size: 0.9rem; color: #333; margin-bottom: 0.5rem; }}
  pre {{ background: #f4f4f4; padding: 0.7rem; border-radius: 6px; overflow-x: auto; font-size: 0.82rem; }}
  pre.diff {{ max-height: 400px; overflow-y: auto; }}
  .actions button {{ padding: 0.4rem 0.9rem; border-radius: 6px; border: 1px solid #ccc;
                     cursor: pointer; margin-right: 0.5rem; font-size: 0.85rem; }}
  .actions .merge {{ background: #1a7f37; color: white; border-color: #1a7f37; }}
  .actions .discard {{ background: white; color: #a00; border-color: #a00; }}
  .empty {{ color: #666; }}
  .recent {{ font-size: 0.85rem; color: #555; }}
  code {{ font-family: ui-monospace, monospace; }}
</style></head>
<body>
<h1>aider-loop review — {esc(project_dir.name)}</h1>
{banner}
{body}
<h2 style="font-size:1rem; margin-top:2rem;">Recently merged</h2>
<ul class="recent">{recent}</ul>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    project_dir: Path  # set on the class before serve_forever()

    def log_message(self, fmt, *args):
        pass  # the terminal running this already shows enough; avoid double noise

    def do_GET(self):
        if self.path not in ("/", ""):
            self.send_response(404)
            self.end_headers()
            return
        self._respond(render_page(self.project_dir))

    def do_POST(self):
        if self.path != "/action":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        fields = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        branch = fields.get("branch", [""])[0]
        action = fields.get("do", [""])[0]

        # Re-validate against the actual branch list rather than trusting
        # the posted name outright - this is a form field, and even on a
        # localhost-only tool a stray or stale request shouldn't be able
        # to name an arbitrary git ref.
        if branch not in list_pending_branches(self.project_dir):
            message = f"'{branch}' is not a pending aider-loop branch (already handled?)"
        elif action == "merge":
            result = git(["merge", "--no-ff", branch, "-m",
                         f"aider-loop review: merge {branch}"], self.project_dir)
            if result.returncode == 0:
                git(["branch", "-D", branch], self.project_dir)
                message = f"Merged {branch} and removed the branch."
            else:
                git(["merge", "--abort"], self.project_dir)
                message = (f"Merge of {branch} failed and was aborted - conflicts with what's "
                           f"changed since. The branch is untouched:\n{result.stdout}{result.stderr}")
        elif action == "discard":
            result = git(["branch", "-D", branch], self.project_dir)
            message = (f"Discarded {branch}." if result.returncode == 0
                      else f"Could not discard {branch}: {result.stderr}")
        else:
            message = f"Unknown action {action!r}"

        self._respond(render_page(self.project_dir, message=message))

    def _respond(self, html_body: str):
        body = html_body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Review aider-loop's parked branches in a browser")
    parser.add_argument("--project-dir", required=True, type=str)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not (project_dir / ".git").exists():
        print(f"{project_dir} is not a git repo.", file=sys.stderr)
        sys.exit(1)

    Handler.project_dir = project_dir
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    pending = len(list_pending_branches(project_dir))
    print(f"Reviewing {project_dir} - {pending} item(s) pending.")
    print(f"Open http://127.0.0.1:{args.port}/ - Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
