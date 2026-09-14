# forge working on itself: the eval suite

The eval.py item merged in one attempt, byte-for-byte, verified against
the real installed model. This second item was rewritten after the
first draft parked - not on a model failure, on my own item-authoring
mistake: it asked for a line-level "Replace it with exactly:" inside an
anchor-based edit, but forge's byte-exact check always compares the
captured spec against the file's ENTIRE content, not a line within it.
The model actually wrote the correct COMMANDS dict and the correct USAGE
line - the item itself was the thing that was wrong. Fixed the only way
that's genuinely safe on a file this size: replace the whole thing.

- [x] Replace the entire contents of `src/forge/cli.py` with exactly:
```python
"""The `forge` command: one entry point, four subcommands.

Deliberately thin - it does not touch argparse in any of the four
modules below, which each already parse `sys.argv` themselves (that's
what let this file exist without risking a single line of change to
code that has already survived real bugs this session). It just picks
the right module and rewrites argv so that module's own parser sees
what it expects: `forge run --project-dir x` becomes the equivalent of
running that module directly with `--project-dir x`.
"""

import sys

USAGE = """\
usage: forge <command> [args]

commands:
  run      Run the unattended loop against a project's checklist
           (was: aider_loop.py)
  draft    Draft checklist items from a goal using a local model
           (was: spec_compiler.py)
  review   Open a local page for reviewing parked branches
           (was: review_server.py)
  eval     Run forge's own regression suite against stub editors

Run `forge <command> --help` for that command's own options.
"""

COMMANDS = {"run": "aider_loop", "draft": "spec_compiler", "review": "review_server", "eval": "eval"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2

    command, rest = argv[0], argv[1:]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"forge: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        return 2

    # Each module's main() calls argparse.parse_args() with no argument,
    # which reads sys.argv[1:] - so argv[0] just needs to be something
    # sensible for the error messages argparse prints (usage: forge-run ...).
    sys.argv = [f"forge-{command}", *rest]
    from importlib import import_module
    module = import_module(f".{module_name}", package="forge")
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] Add unit tests in `tests/test_lite_editor.py` for the
  find_import_context function (it currently has none). Use tmp_path or
  a similar temp directory as the fake project root, writing small
  Python files into it directly rather than depending on any other
  project's fixtures. Cover three cases: (1) a target file whose own
  import statement resolves to a real file already on disk in the temp
  project, (2) a target file that does not exist yet, where a resolvable
  reference only appears in the item text argument (not in any file on
  disk) - this is the case TDD actually depends on, since both files
  being written are new, and (3) a reference to something that is not a
  real file in the temp project (e.g. a stdlib-style name) being
  silently skipped rather than raising or being included in the result.
