"""Syntax validation, preflight scans, and file-level integrity checks for forge."""

import fnmatch
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_MAX_FILE_BYTES, EXCLUDE_DIRS, cfg, log

# File paths referenced in backticks in a task's text, e.g. "In `src/foo.jsx`,
# do X" - used to sanity-check that aider actually touched the file(s) the
# task named, rather than trusting "it built" alone.
# Any extension, as long as it starts with a letter (so `3.11` isn't a file).
# This used to be a fixed list of extensions, which was harmless while it only
# fed the touched-files check -- but restore_unnamed_files() reverts whatever
# isn't named, so an unlisted extension (a project's own `.story` files) would
# have had each task's real output deleted.
FILE_PATH_RE = re.compile(r"`([\w./-]*[\w-]\.[A-Za-z][A-Za-z0-9]{0,7})`")

# Extensions common enough to trust even on a name that looks like code.
# NOT a whitelist - an unlisted extension is still a file unless the name
# also has code shape (see _looks_like_code_reference).
_COMMON_EXTENSIONS = {
    "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "json", "md", "txt", "toml", "yaml",
    "yml", "ini", "cfg", "html", "css", "scss", "sh", "rs", "go", "java", "kt", "c",
    "h", "cpp", "hpp", "cs", "rb", "php", "lua", "sql", "xml", "csv", "lock", "env",
}


def _looks_like_code_reference(path: str) -> bool:
    """True for backticked code like `Ledger.add` or `ledger.store.Ledger`,
    which FILE_PATH_RE also matches. Found by forge calibrate: a rename
    item was parked on every trial for never "touching" `Ledger.add`, and
    a TDD item naming `ledger.store.Ledger` was rejected before any model
    ran for naming three files instead of two.

    Only slash-less names qualify, and only with a code shape: a
    capitalised stem or extension, or more than one dot. A real file
    written that way with an uncommon extension would be missed, which is
    the smaller risk than parking correct work."""
    if "/" in path:
        return False
    stem, _, ext = path.rpartition(".")
    if ext.lower() in _COMMON_EXTENSIONS and not ext[0].isupper():
        return False
    return stem[:1].isupper() or ext[:1].isupper() or "." in stem.lstrip(".")


def find_js_html_files(project_dir: Path) -> list[Path]:
    exts = {".js", ".html", ".mjs"}
    return [p for p in project_dir.rglob("*")
            if p.suffix in exts and not (EXCLUDE_DIRS & set(p.parts))]


def find_python_files(project_dir: Path) -> list[Path]:
    return [p for p in project_dir.rglob("*.py") if not (EXCLUDE_DIRS & set(p.parts))]


def run_configured_commands(project_dir: Path, log_path: Path) -> tuple[bool, str]:
    """Run whatever `[validate] commands` the project declared, in order.

    This replaces the hardcoded pytest invocation that used to live here as
    FAST_TEST_ARGS. A test suite is a project-wide gate and SHOULD run in
    full -- unlike the syntax checks below, which are scoped to the files an
    item actually touched. Failing tests in an untouched module are real
    signal that the change broke something; a syntax error in an untouched
    file is not.

    Commands run with cwd=project_dir through the shell, so a project can
    write ".venv/bin/pytest tests/fast -q" or "npm test --silent" without
    this script knowing anything about either.
    """
    commands = cfg("validate", "commands", default=[]) or []
    problems = []
    for command in commands:
        try:
            result = subprocess.run(command, shell=True, cwd=str(project_dir),
                                    capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as e:
            # A misconfigured or hanging command should degrade this check,
            # not kill an unattended run.
            log(f"validate command failed to run (skipping): {command}: {e}", log_path)
            continue
        if result.returncode != 0:
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            problems.append(f"$ {command}\n{output[-4000:]}")
    if problems:
        msg = "Project validate command(s) failed:\n\n" + "\n\n".join(problems)
        log(msg[:2000], log_path)
        return False, msg
    if commands:
        log(f"Validation passed ({len(commands)} project command(s))", log_path)
    return True, "ok"


def validate_python(project_dir: Path, log_path: Path,
                    scope: list[Path] | None = None) -> tuple[bool, str]:
    """compile() every .py file in `scope` (default: all of them).

    `scope` is the list of files the item actually changed. Checking only
    those is the important part, not an optimisation: a syntax check that
    walks the whole repo will fail an item for a problem in a file it never
    opened, and the loop then feeds that failure back as "fix the failure
    above while still completing the original task" -- pointing a whole-file
    edit model at unrelated code. That is not hypothetical; it happened, and
    it is why this takes a scope argument at all.

    Deliberately NOT relying on aider's own --auto-test/--auto-lint: that is
    not reliably applied in the headless `--message` + `--yes-always` mode
    this script uses (see Aider-AI/aider#4923). This is the safety net, not
    a backstop for one that might not be running.
    """
    files = [f for f in (scope if scope is not None else find_python_files(project_dir))
             if f.suffix == ".py" and f.is_file()]
    problems = []
    for f in files:
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except SyntaxError as e:
            problems.append(f"{f}:{e.lineno}: {e.msg}")
        except OSError:
            continue

    if problems:
        msg = "Python syntax errors:\n" + "\n".join(problems)
        log(msg, log_path)
        return False, msg

    log(f"Validation passed (python syntax, {len(files)} file(s))", log_path)
    return True, "ok"


# Server-side template syntax that is NOT JavaScript and must be removed
# before an inline <script> body can be handed to `node --check`. Without
# this, every Jinja template in a Flask project fails validation on its
# first `{{ ... }}`, which has nothing to do with whatever the model just
# edited: a one-line change to a .py file was reported as a SyntaxError in
# running.html, and the loop then asked the model to "fix" a template it had
# never touched. Stripping rather than skipping keeps real coverage -- a
# genuine JS error in a template's inline script is still caught.
_JINJA_EXPR_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)      # {{ value }} -> a literal
_JINJA_STMT_RE = re.compile(r"\{%.*?%\}", re.DOTALL)          # {% if %}    -> nothing
_JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", re.DOTALL)       # {# note #}  -> nothing


def strip_template_syntax(script_body: str) -> str:
    """Replace Jinja/Django expressions with a JS literal and drop control
    tags, so what's left is checkable JavaScript.

    `{{ ... }}` becomes `null` because it almost always appears in value
    position (`var runId = {{ run_id | tojson }};`), where deleting it would
    create a syntax error of our own making. Control tags are deleted
    outright: the statements they wrap are themselves ordinary JS and stay
    valid without the surrounding tag.
    """
    out = _JINJA_COMMENT_RE.sub("", script_body)
    out = _JINJA_STMT_RE.sub("", out)
    return _JINJA_EXPR_RE.sub("null", out)


def validate_js_html(project_dir: Path, log_path: Path,
                     scope: list[Path] | None = None) -> tuple[bool, str]:
    """
    Syntax/lint-level validation only, per your call - this does NOT run the
    game or check gameplay correctness, just confirms nothing is outright
    broken (JS parse errors, obviously malformed HTML).

    If project_dir has a package.json with a "build" script, that build is
    run instead of the per-file checks below - node --check can't parse JSX,
    so a project full of .jsx files would otherwise get zero real validation
    (every file silently skipped) while still being reported as "passed".
    Running the actual build catches JSX syntax errors correctly and is the
    only way to validate this project.

    Otherwise, prefers node --check for .js files if node is available.
    Falls back to a lightweight HTML sanity check (balanced <script> tags,
    no obvious truncation) since a full HTML validator isn't worth the
    dependency here.
    """
    package_json = project_dir / "package.json"
    # Only run a full build when the project asked for it; "package.json
    # exists" is not consent to run npm on every item.
    if "npm-build" in (cfg("validate", "checks", default=[]) or []) and package_json.is_file():
        try:
            has_build_script = "build" in json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except Exception:
            has_build_script = False
        if has_build_script:
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=str(project_dir), capture_output=True, text=True,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                msg = f"npm run build failed:\n{output}"
                log(msg, log_path)
                return False, msg
            log("Validation passed (npm run build)", log_path)
            return True, "ok"

    problems = []

    node_available = subprocess.run(
        ["which", "node"], capture_output=True, text=True
    ).returncode == 0

    candidates = scope if scope is not None else find_js_html_files(project_dir)
    for f in [c for c in candidates if c.suffix in (".js", ".mjs", ".html") and c.is_file()]:
        if f.suffix in (".js", ".mjs"):
            if node_available:
                result = subprocess.run(
                    ["node", "--check", str(f)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    problems.append(f"{f}: {result.stderr.strip()}")
            # if node isn't available, we silently skip JS syntax checking
            # rather than failing the whole run over a missing tool
        elif f.suffix == ".html":
            content = f.read_text(encoding="utf-8", errors="replace")
            if content.count("<script") != content.count("</script>"):
                problems.append(f"{f}: mismatched <script> tags")
            # extract inline <script> blocks and check those too, if node available
            if node_available:
                for match in re.finditer(
                    r"<script(?:\s[^>]*)?>(.*?)</script>", content, re.DOTALL
                ):
                    script_body = strip_template_syntax(match.group(1))
                    if not script_body.strip():
                        continue
                    # Written to the system temp dir, not project_dir: a stray
                    # .js file inside the repo is something find_js_html_files
                    # would then try to validate, and something
                    # suspicious_new_paths would reasonably flag as the model
                    # having invented a file.
                    fd, tmp_name = tempfile.mkstemp(prefix="aider_inline_", suffix=".js")
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as fh:
                            fh.write(script_body)
                        result = subprocess.run(
                            ["node", "--check", tmp_name],
                            capture_output=True, text=True,
                        )
                    finally:
                        try:
                            os.unlink(tmp_name)
                        except OSError:
                            pass
                    if result.returncode != 0:
                        problems.append(f"{f} (inline script): {result.stderr.strip()}")

    if problems:
        msg = "Validation found issues:\n" + "\n".join(problems)
        log(msg, log_path)
        return False, msg

    log("Validation passed (syntax-level check only)", log_path)
    return True, "ok"


def check_syntax_only(project_dir: Path, log_path: Path,
                      scope: list[Path] | None = None) -> list[tuple[bool, str]]:
    """The syntax-checker half of validate_syntax(), without the project's
    own `[validate] commands`.

    Split out because those two things answer different questions: "is
    this file well-formed" and "does the project's test suite pass" are
    not the same check, and TDD's red phase needs only the first - a
    crash in the new test isn't "a failing test" and shouldn't be treated
    as one, but the project's test command failing IS exactly what red
    phase expects and must not be mistaken for a syntax problem either.
    Calling validate_syntax() itself for this would run the project's
    commands a phase early and misreport the result: measured directly,
    the red-phase test importing a not-yet-written module made the
    project's own `commands` gate fail (correctly - that's the point of
    red phase), and validate_syntax() folded that into "syntax error",
    which it wasn't.
    """
    checks = cfg("validate", "checks", default=None)
    if checks is None:
        checks = ["python", "js-html"]

    results = []
    if "python" in checks:
        results.append(validate_python(project_dir, log_path, scope))
    if "js-html" in checks or "npm-build" in checks:
        results.append(validate_js_html(project_dir, log_path, scope))
    return results


def validate_syntax(project_dir: Path, log_path: Path,
                    scope: list[Path] | None = None) -> tuple[bool, str]:
    """Run the built-in syntax checkers that apply, over `scope` only.

    Which checkers run comes from `[validate] checks` in .aiderloop.toml
    ("python", "js-html", "npm-build"). With no config, both syntax
    checkers are enabled and each no-ops when the changed set contains no
    files it understands -- so a brand-new project gets sensible behaviour
    before anyone writes config, and a project that wants something
    narrower can say so.

    `scope` is the set of files the item changed. See validate_python's
    docstring for why that scoping is load-bearing rather than an
    optimisation.
    """
    results = check_syntax_only(project_dir, log_path, scope)
    results.append(run_configured_commands(project_dir, log_path))

    ok = all(r[0] for r in results)
    combined = "\n\n".join(r[1] for r in results if r[1] and r[1] != "ok")
    return ok, combined or "ok"


def package_json_or_js_html_present(project_dir: Path) -> bool:
    return (project_dir / "package.json").is_file() or bool(find_js_html_files(project_dir))


def preflight_repo_size(project_dir: Path, log_path: Path) -> list[str]:
    """Warn about files big enough to wreck aider's context window.

    Aider builds its repo map by walking the working tree, so a few
    committed data files can dominate everything. Measured on a real
    project: ~3MB of golden CSV fixtures and a 6MB generated HTML report
    produced an estimated chat context of 1,584,443 tokens against a 32,768
    limit, for a task that touched one template -- and killed two runs by
    exhausting system memory before anyone understood why.

    The fix is a `.aiderignore` listing that data. This check exists so the
    next project finds that out in the first ten seconds instead of after
    two OOM kills, so it names the offenders and the file to put them in.
    Warn-only: a big file is a strong smell, not proof of a problem, and
    refusing to start would be the wrong call on a project that genuinely
    needs one.
    """
    limit = cfg("preflight", "max_file_bytes", default=DEFAULT_MAX_FILE_BYTES)
    ignore_path = project_dir / ".aiderignore"
    ignored = []
    if ignore_path.is_file():
        ignored = [l.strip() for l in ignore_path.read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.strip().startswith("#")]

    def is_ignored(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pat) or rel.startswith(pat.rstrip("/") + "/")
                   for pat in ignored)

    big = []
    for f in project_dir.rglob("*"):
        if not f.is_file() or (EXCLUDE_DIRS & set(f.parts)) or ".git" in f.parts:
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size <= limit:
            continue
        rel = str(f.relative_to(project_dir))
        if not is_ignored(rel):
            big.append(f"{rel} ({size // 1024}KB)")

    if big:
        log(f"Preflight: {len(big)} file(s) over {limit // 1024}KB are visible to aider "
            f"and not in .aiderignore. These inflate every prompt and can exhaust the "
            f"context window:\n  " + "\n  ".join(sorted(big)[:15]), log_path)
    return big


# Phrases that, when they appear shortly after a backtick-quoted file path,
# mean that file was named as a "don't touch this" reference (e.g. "...inside
# `src/foo.jsx`, which is unrelated and should not be touched") rather than a
# file the task expects the commit to actually change. Found by testing this
# exact check against a real task and getting a false positive on exactly
# this phrasing.
# The (?!...) after "touch" stops "do not touch any other file"/"...other
# files" from matching: that phrase means "nothing ELSE should be edited",
# which is the opposite of negating the file just named - it showed up as
# this project's own boilerplate scoping instruction ("Do not touch any
# other file.") on every exact-content-spec task, and without this
# exclusion it silently negated that same task's real, single expected
# file, making expected_files() return empty for every one of them. Found
# by writing the exact-content-spec check and getting an empty result on a
# real task's real target file.
NEGATION_NEAR_RE = re.compile(
    r"should not be touched|do not touch(?!\s+(?:any\s+)?other)|not be touched|"
    r"should not touch(?!\s+(?:any\s+)?other)|"
    # "change" is a much more generic verb than "touch" in practice - real
    # items use "do not change any existing test" as a plain disclaimer
    # with no specific file in view. A backtick-lookahead here doesn't
    # work as a fix (tried first, reverted): _negates_before's `before`
    # slice deliberately stops right BEFORE a path's own backtick, so a
    # lookahead requiring one immediately after can never match there -
    # that's the whole call site this exists for. Same exclusion shape as
    # "touch" instead, widened to the two boilerplate phrasings actually
    # seen ("any other ..." / "any existing ...").
    r"do not change(?!\s+(?:any\s+)?(?:other|existing))|"
    r"should not change(?!\s+(?:any\s+)?(?:other|existing))|"
    r"leave\b.{0,40}\bunchanged|which is unrelated",
    re.IGNORECASE,
)


def path_matches_any(expected_path: str, touched_paths: list[str]) -> bool:
    """
    True if expected_path refers to the same file as one of touched_paths,
    allowing for the same file being written two different ways (a full
    repo-relative path like "src/App.jsx" vs. a bare filename like "App.jsx"
    used as shorthand later in the same task's text) - git diff always
    reports the full path, so a literal-string comparison would otherwise
    flag a correctly-touched file as missing. Found via a real task that did
    exactly this in its own wording.
    """
    for touched in touched_paths:
        if expected_path == touched:
            return True
        if touched.endswith("/" + expected_path) or expected_path.endswith("/" + touched):
            return True
    return False


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.DOTALL)


def _negates_before(before: str) -> bool:
    """True if a "don't touch this" phrase sits immediately BEFORE a path,
    with no other path in between - i.e. this path is the one it refers to.

    _negates() below only ever looked at text AFTER a path, which meant it
    caught "`bar.py`, which is unrelated" but silently missed the far more
    natural "do not touch `bar.py`" whenever that path was the last one in
    the item. Measured cost of that gap: on archaeologist's first full
    run, twelve of twenty parked items were items whose own text ended
    "...and do not touch `arch/chunk.py`." The loop dutifully required the
    commit to touch the very file the item forbade touching, and parked
    every one of them for not doing so.
    """
    last = None
    for m in NEGATION_NEAR_RE.finditer(before):
        last = m  # nearest phrase to this path wins
    if last is None:
        return False
    # Only this path is negated - if another path appears between the
    # phrase and here, that one was its target, not this one.
    return not FILE_PATH_RE.search(before[last.end():])


def _negates(path: str, after: str) -> bool:
    """True if `after` (the text following one mention of `path`) contains
    a negation phrase that is actually about `path`, not some other file
    also named nearby. NEGATION_NEAR_RE's phrases point in different
    directions - "do not touch `other.py`" names its file AFTER the
    phrase, "`other.py`, which ... should not be touched" names it
    BEFORE - so which file a match belongs to has to be resolved by
    proximity to the phrase itself, not just by finding some negation
    phrase anywhere in the window. Without this, a task like "In `foo.py`,
    add X. Do not touch `bar.py`, which is unrelated." would negate foo.py
    (an earlier, different file) purely for sharing a window with bar.py's
    own, correctly-targeted negation."""
    neg = NEGATION_NEAR_RE.search(after)
    if not neg:
        return False
    before_neg = after[:neg.start()]
    after_neg = after[neg.end():neg.end() + 60]
    # Forward-referring ("do not touch `other.py`"): whatever's named
    # right after the phrase is what's negated, if anything is.
    forward = FILE_PATH_RE.search(after_neg)
    if forward:
        return forward.group(1) == path
    # Backward-referring ("`other.py`, which ... should not be touched"):
    # only relevant if some path was actually named between this mention
    # and the phrase - if one was, that's what's negated instead of path.
    backward = None
    for pm in FILE_PATH_RE.finditer(before_neg):
        backward = pm.group(1)  # last match wins - nearest to the phrase
    if backward is not None:
        return backward == path
    # Nothing named on either side: the phrase directly follows this
    # mention with nothing else in between - the original, well-tested
    # case ("...inside `src/foo.jsx`, which is unrelated and should not
    # be touched").
    return True


def expected_files(item_text: str) -> list[str]:
    """Backtick-quoted file paths mentioned in the task text (outside any
    fenced code block) - the files we expect this item's commit to actually
    touch. A path counts unless a "don't touch this" phrase follows it
    nearby (NEGATION_NEAR_RE) - see the long comment above NEGATION_NEAR_RE's
    old sibling regex for why this isn't also gated on an edit-action verb
    preceding it.

    Fenced blocks are stripped before scanning: since parse_todo() started
    capturing an item's full multi-line text (see its docstring) rather
    than just the checkbox line, a task that spells out a whole target
    file's content inline - which is most of them, in this project - has
    that content's OWN file references show up in item_text too (e.g. a
    README documenting "site.yaml" and "fabric.yaml" in its own prose).
    Those aren't edit targets, they're the target file's content; only
    prose outside any fence names a real one ("In `results.html`, add...").
    extract_exact_content_specs() is what actually verifies fenced content,
    separately and more precisely than this function ever could."""
    prose_only = _FENCE_RE.sub("", item_text)
    expected = []
    for m in FILE_PATH_RE.finditer(prose_only):
        after = prose_only[m.end():m.end() + 120]
        before = prose_only[max(0, m.start() - 120):m.start()]
        if _looks_like_code_reference(m.group(1)):
            continue
        if _negates_before(before) or _negates(m.group(1), after):
            continue
        expected.append(m.group(1))
    return sorted(set(expected))


PROMPT_LEAKAGE_MARKERS = (
    "Plan the change, then implement it",
    "ONLY EVER RETURN CODE IN A SEARCH/REPLACE BLOCK",
    "I added these files to the chat",
    "<... existing content ...>",
    "... existing content ...",
    "... existing code ...",
    "... rest of code ...",
    "... rest of the code ...",
    "... remaining code ...",
    "/* ... existing code ... */",
    "// ... existing code ...",
    "<!-- ... existing code ... -->",
)


def detect_prompt_leakage(project_dir: Path, touched_paths: list[str]) -> list[str]:
    """Returns the subset of touched_paths whose current on-disk content
    contains a known prompt-leakage marker verbatim."""
    corrupted = []
    for rel in touched_paths:
        path = project_dir / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker in text for marker in PROMPT_LEAKAGE_MARKERS):
            corrupted.append(rel)
    return corrupted


_WINDOWS_ILLEGAL_CHARS_RE = re.compile(r'[<>:"|?*]')


def suspicious_new_paths(touched_paths: list[str]) -> list[str]:
    """Returns the subset of touched_paths that look like a garbage
    filename rather than a real one aider meant to create: a Windows-illegal
    character, or implausibly long for a single path component (a real
    source/doc filename is never this long)."""
    suspicious = []
    for rel in touched_paths:
        basename = rel.rsplit("/", 1)[-1]
        if _WINDOWS_ILLEGAL_CHARS_RE.search(rel) or len(basename) > 100:
            suspicious.append(rel)
    return suspicious


def extract_exact_content_specs(item_text: str) -> dict[str, str]:
    """Many tasks in this project's todo files spell out a file's target
    content verbatim: "...`path/to/file` with exactly this content:" or
    "...`path/to/file` with exactly:", immediately followed by a fenced
    code block. When a task is phrased that way, there's no need to guess
    whether what aider wrote matches -- it can be checked directly, byte
    for byte (mod trailing-newline normalization), rather than inferred
    from heuristics like expected_files() or detect_prompt_leakage(). This
    only covers tasks actually phrased this way; anything else still falls
    back to those.

    Returns {file_path: expected_content} for every such spec found."""
    specs: dict[str, str] = {}
    for fence in _FENCE_RE.finditer(item_text):
        before = item_text[:fence.start()].rstrip()
        if not before.endswith(":"):
            continue
        if "exactly" not in before[-80:].lower():
            continue
        path_matches = [m for m in FILE_PATH_RE.finditer(item_text[:fence.start()])
                        if not _looks_like_code_reference(m.group(1))]
        if not path_matches:
            continue
        specs[path_matches[-1].group(1)] = fence.group(1)
    return specs


def _normalize_content(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n")


def content_mismatches(project_dir: Path, specs: dict[str, str]) -> list[str]:
    """Returns the subset of specs' file paths whose actual on-disk content
    doesn't match the spec (after normalizing line endings and a trailing
    newline, since a real file's own convention there is irrelevant to
    whether the meaningful content is right)."""
    mismatched = []
    for rel, expected in specs.items():
        path = project_dir / rel
        try:
            actual = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            mismatched.append(rel)  # doesn't exist at all -- also a mismatch
            continue
        if _normalize_content(actual) != _normalize_content(expected):
            mismatched.append(rel)
    return mismatched


def changed_files(project_dir: Path, pre_hash: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", pre_hash, "HEAD"],
        cwd=str(project_dir), capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def restore_unnamed_files(project_dir: Path, pre_hash: str, expected: list[str],
                          log_path: Path) -> list[str]:
    """Reverts every file changed since pre_hash that the task didn't name,
    commits the restoration, and returns the paths restored. Does nothing
    when the task named no files (there's no scope to enforce) or when the
    project sets [scope] restore_unnamed = false."""
    if not expected or not cfg("scope", "restore_unnamed", default=True):
        return []
    extra = [f for f in changed_files(project_dir, pre_hash)
             if not any(path_matches_any(e, [f]) for e in expected)]
    if not extra:
        return []
    for rel in extra:
        existed = subprocess.run(["git", "cat-file", "-e", f"{pre_hash}:{rel}"],
                                 cwd=str(project_dir), capture_output=True).returncode == 0
        if existed:
            subprocess.run(["git", "checkout", pre_hash, "--", rel],
                           cwd=str(project_dir), capture_output=True)
        else:
            subprocess.run(["git", "rm", "-q", "-f", "--", rel],
                           cwd=str(project_dir), capture_output=True)
    subprocess.run(["git", "add", "-A", "--", *extra], cwd=str(project_dir), capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"aider-loop: restore files the task didn't name: {', '.join(extra)}"],
                   cwd=str(project_dir), capture_output=True)
    log(f"Restored file(s) the task didn't name: {extra} (task named: {expected})", log_path)
    return extra
