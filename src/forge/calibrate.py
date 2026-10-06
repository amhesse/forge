"""Measures which kinds of item each model can be trusted with, so routing
between the strong and weak worker tiers rests on data instead of on
estimate_difficulty()'s one untested assumption.

Every (model, item, trial) runs alone: a fresh git copy of
bench/fixture/, a one-item checklist, and the real `forge run` as a
subprocess with that single model. Afterwards a hidden grader
(bench/graders/<id>.py, never visible to the model) is run against the
result, and against the parked branch too if forge didn't merge it. That
separates four outcomes forge alone can't tell apart:

  pass           merged, and actually correct
  silent_wrong   merged, but wrong - the safety net missed it. The only
                 outcome that makes a model unsafe for a category.
  parked_ok      parked, though the work was correct - checks too strict
  parked         parked, and wrong - the safety net did its job

Results append to a JSONL file one trial at a time, so a long overnight
run can be killed and resumed without redoing finished trials.
"""

import argparse
import collections
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from . import aider_loop

DEFAULT_BENCH = Path(__file__).resolve().parents[2] / "bench"
OUTCOMES = ["pass", "silent_wrong", "parked_ok", "parked"]
# A category is "safe for the weak tier" at this pass rate with zero
# silent_wrong. Parks only cost time; a silent wrong answer costs trust.
SAFE_PASS_RATE = 0.8


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.email=bench@forge", "-c", "user.name=bench", *args],
                          cwd=cwd, capture_output=True, text=True, check=check)


def signals(text: str) -> dict:
    """The routing signals forge itself can see in an item's text."""
    return {
        "tdd": aider_loop.is_tdd_item(text),
        "files": len(aider_loop.expected_files(text)),
        "exact": bool(aider_loop.extract_exact_content_specs(text)),
        "forge_tier": aider_loop.estimate_difficulty(text),
    }


def grade(checkout: Path, grader: Path) -> tuple[bool, str]:
    """Hidden grader plus the fixture's own visible tests; both must pass."""
    target = checkout / "_forge_grader.py"
    shutil.copy(grader, target)
    try:
        for cmd in ([sys.executable, target.name],
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"]):
            r = subprocess.run(cmd, cwd=checkout, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return False, (r.stdout + r.stderr)[-1500:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "grader timed out"
    finally:
        target.unlink(missing_ok=True)


def run_trial(bench: Path, item: dict, model: str, backend: str, timeout: int) -> dict:
    name = f"forge-bench-{item['id']}-{os.getpid()}-{time.time_ns()}"
    cache = Path.home() / ".cache" / "aider-loop" / name
    with tempfile.TemporaryDirectory() as d:
        proj = Path(d) / name
        shutil.copytree(bench / "fixture", proj)
        (proj / "TODO.md").write_text(f"- [ ] {item['text'].strip()}\n")
        git(proj, "init", "-q", "-b", "master")
        git(proj, "add", "-A")
        git(proj, "commit", "-q", "-m", "fixture")

        cmd = ["forge", "run", "--project-dir", str(proj), "--todo-file", "TODO.md",
               "--models", model, "--backend", backend, "--sleep-between", "0"]
        started = time.monotonic()
        try:
            r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True, timeout=timeout)
            log = r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            log = f"TIMEOUT after {timeout}s\n{e.stdout or ''}"
        seconds = round(time.monotonic() - started, 1)

        todo = (proj / "TODO.md").read_text()
        marker = next((m for m in ("[x]", "[!]", "[?]") if f"- {m}" in todo), "[ ]")
        grader = bench / "graders" / f"{item['id']}.py"
        merged = marker == "[x]"
        if merged:
            correct, detail = grade(proj, grader)
        else:
            branches = git(proj, "branch", "--list", "aider-loop/*", "--format=%(refname:short)").stdout.split()
            if branches:
                git(proj, "checkout", "-q", "-f", branches[-1])
                correct, detail = grade(proj, grader)
            else:
                correct, detail = False, "no branch left to grade"
        outcome = ("pass" if correct else "silent_wrong") if merged else ("parked_ok" if correct else "parked")

        tokens = {}
        for rec in sorted((cache / "runs").glob("*/item-*.json")):
            try:
                data = json.loads(rec.read_text())
            except ValueError:
                continue
            for k, v in (data.get("tokens") or {}).items():
                if isinstance(v, (int, float)):
                    tokens[k] = tokens.get(k, 0) + v
    shutil.rmtree(cache, ignore_errors=True)

    return {
        "model": model, "backend": backend, "item": item["id"], "category": item["category"],
        **signals(item["text"]), "marker": marker, "outcome": outcome, "seconds": seconds,
        "tokens": tokens, "grader_output": detail, "forge_log_tail": log[-2000:],
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(results: list[dict]) -> str:
    if not results:
        return "no results yet"
    out = []
    models = sorted({r["model"] for r in results})
    for key, label in (("category", "category"), ("item", "item")):
        groups = sorted({r[key] for r in results})
        width = max(len(label), *(len(g) for g in groups))
        out.append(f"\nBy {label}  (pass / silent_wrong / parked_ok / parked,  median seconds)")
        out.append(f"{'':{width}}  " + "  ".join(f"{m:>28}" for m in models))
        for g in groups:
            cells = []
            for m in models:
                rs = [r for r in results if r["model"] == m and r[key] == g]
                if not rs:
                    cells.append(f"{'-':>28}")
                    continue
                c = collections.Counter(r["outcome"] for r in rs)
                secs = sorted(r["seconds"] for r in rs)[len(rs) // 2]
                cells.append(f"{'/'.join(str(c[o]) for o in OUTCOMES):>18} {secs:>7.0f}s ")
            out.append(f"{g:{width}}  " + "  ".join(cells))

    out.append(f"\nSafe per model (pass rate >= {SAFE_PASS_RATE:.0%} and zero silent_wrong):")
    for m in models:
        safe, unsafe = [], []
        for cat in sorted({r["category"] for r in results}):
            rs = [r for r in results if r["model"] == m and r["category"] == cat]
            if not rs:
                continue
            c = collections.Counter(r["outcome"] for r in rs)
            ok = c["silent_wrong"] == 0 and c["pass"] / len(rs) >= SAFE_PASS_RATE
            (safe if ok else unsafe).append(f"{cat} ({c['pass']}/{len(rs)})")
        out.append(f"  {m}\n    safe:   {', '.join(safe) or '-'}\n    unsafe: {', '.join(unsafe) or '-'}")
    wrong = [r for r in results if r["outcome"] == "silent_wrong"]
    if wrong:
        out.append("\nSilent wrong answers (forge merged work the grader rejected):")
        out.extend(f"  {r['model']}  {r['item']}" for r in wrong)
    return "\n".join(out)


def self_test(bench: Path) -> int:
    """Checks the bench itself: each grader must fail on the untouched
    fixture and pass on its reference solution. No model involved."""
    items = tomllib.loads((bench / "items.toml").read_text())["item"]
    bad = 0
    for item in items:
        grader = bench / "graders" / f"{item['id']}.py"
        with tempfile.TemporaryDirectory() as d:
            before = Path(d) / "before"
            shutil.copytree(bench / "fixture", before)
            fails_before = not grade(before, grader)[0]
            after = Path(d) / "after"
            shutil.copytree(bench / "fixture", after)
            shutil.copytree(bench / "reference" / item["id"], after, dirs_exist_ok=True)
            passes_after, detail = grade(after, grader)
        ok = fails_before and passes_after
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {item['id']}"
              + ("" if fails_before else "  (grader passes on untouched fixture)")
              + ("" if passes_after else f"  (reference fails: {detail.strip()[-300:]})"))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forge calibrate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", help="Comma-separated ollama models, e.g. qwen2.5-coder:14b,qwen2.5-coder:7b")
    parser.add_argument("--trials", type=int, default=3, help="Runs per model per item (default 3)")
    parser.add_argument("--backend", choices=["aider", "lite"], default="lite")
    parser.add_argument("--items", help="Comma-separated item ids to run (default: all)")
    parser.add_argument("--out", default="calibration.jsonl", help="Results file; appended to and resumed from")
    parser.add_argument("--timeout", type=int, default=45 * 60, help="Seconds per trial")
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--report", action="store_true", help="Only print the report for --out")
    parser.add_argument("--self-test", action="store_true", help="Verify graders against reference solutions")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not args.report and not (args.bench_dir / "items.toml").is_file():
        # The bench is kept out of the public repository: graders a model
        # could have trained on would stop being hidden.
        parser.error(f"no calibration bench at {args.bench_dir} (items.toml not found). The "
                     f"bench is not distributed with forge; pass --bench-dir to your own, laid "
                     f"out as README's 'Measuring it: forge calibrate' describes.")
    if args.self_test:
        return self_test(args.bench_dir)
    if args.report:
        print(report(load_results(out)))
        return 0
    if not args.models:
        parser.error("--models is required (unless --report or --self-test)")

    items = tomllib.loads((args.bench_dir / "items.toml").read_text())["item"]
    if args.items:
        wanted = set(args.items.split(","))
        items = [i for i in items if i["id"] in wanted]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    done = collections.Counter((r["model"], r["backend"], r["item"]) for r in load_results(out))
    # Interleave models and items so a run stopped early still has a bit of everything.
    plan = [(t, m, i) for t in range(args.trials) for i in items for m in models
            if done[(m, args.backend, i["id"])] <= t]
    print(f"{len(plan)} trial(s) to run, results -> {out}")
    for n, (t, model, item) in enumerate(plan, 1):
        print(f"[{n}/{len(plan)}] {model}  {item['id']}  trial {t + 1} ...", end=" ", flush=True)
        result = run_trial(args.bench_dir, item, model, args.backend, args.timeout)
        with out.open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"{result['outcome']} ({result['seconds']:.0f}s)")
    print(report(load_results(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
