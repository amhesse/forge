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

Run `forge <command> --help` for that command's own options.
"""

COMMANDS = {"run": "aider_loop", "draft": "spec_compiler", "review": "review_server"}


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
