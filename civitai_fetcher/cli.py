"""
DEPRECATED — this module is a placeholder, not the working CLI.

`civitai_fetcher.cli` used to import a `fetch_all` from fetch.py that was
removed in commit 1df2185 without repointing this file — that's why it's
been raising ImportError since. Rather than resurrect that exact name, the
image-fetching pipeline now lives in `civitai_fetcher.images_cli`, built on
top of the split client.py / activity.py / images.py modules (see each
module's docstring for why they're separated).

Use:
    uv run python -m civitai_fetcher.images_cli
"""
import sys


def main():
    print(
        "civitai_fetcher.cli is deprecated and no longer implemented.\n"
        "Use: uv run python -m civitai_fetcher.images_cli\n"
        "(run with --help to see all options)",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()