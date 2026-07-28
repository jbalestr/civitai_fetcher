"""
Single entrypoint for every civitai-fetcher command, replacing five separate
`python -m civitai_fetcher.X_cli` invocations with one dispatcher:

    civitai-fetcher images   ...   (was images_cli.py)
    civitai-fetcher models   ...   (was model_cli.py)
    civitai-fetcher creators ...   (was creator_cli.py)
    civitai-fetcher patterns ...   (was patterns_cli.py)
    civitai-fetcher probe    ...   (was probe.py)

Each subcommand still owns its full argparse surface (help text, defaults,
suppressed/advanced flags) — this dispatcher only picks which one runs, it
doesn't touch their argument parsing.
"""
import sys

from .commands import images, models, creators, patterns
from ..services import probe

COMMANDS = {
    "images": images.main,
    "models": models.main,
    "creators": creators.main,
    "patterns": patterns.main,
    "probe": probe.main,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        names = ", ".join(COMMANDS)
        print(f"Usage: civitai-fetcher <command> [args...]\nCommands: {names}")
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    command = sys.argv[1]
    # Strip the subcommand name so each command's own argparse sees argv as
    # if it were invoked directly, e.g. `civitai-fetcher models --model-id 1`
    # -> models.main() sees argv = ["--model-id", "1"].
    sys.argv = [f"civitai-fetcher {command}"] + sys.argv[2:]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
