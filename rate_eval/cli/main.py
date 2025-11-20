"""Main CLI entry point for rate_eval package."""

import argparse
import contextlib
import sys
from typing import Callable, Iterable, List, Optional

from .extract import extract_embeddings_cli
from .evaluate import evaluate_embeddings_cli
from ..common import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rate-eval",
        description="RATE Evaluation Pipeline - Extract and evaluate embeddings from vision-language models",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=False)

    def add_command(name: str, func: Callable[[], None], help_text: str) -> None:
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.set_defaults(handler=func, command_name=name)

    add_command("extract", extract_embeddings_cli, "Extract embeddings from models")
    add_command("evaluate", evaluate_embeddings_cli, "Evaluate embeddings using cached features")

    return parser


@contextlib.contextmanager
def _temporary_argv(argv: Iterable[str]) -> None:
    original = sys.argv[:]
    try:
        sys.argv = list(argv)
        yield
    finally:
        sys.argv = original


def _dispatch(
    parser: argparse.ArgumentParser, parsed_args: argparse.Namespace, remainder: List[str]
) -> int:
    handler: Optional[Callable[[], None]] = getattr(parsed_args, "handler", None)
    command_name: Optional[str] = getattr(parsed_args, "command_name", None)

    if handler is None or command_name is None:
        parser.print_help()
        return 1

    argv = [f"{parser.prog} {command_name}".strip()] + remainder

    try:
        with _temporary_argv(argv):
            handler()
        return 0
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception:
        logger.exception("Command '%s' failed", command_name)
        return 1


def main(args: Optional[Iterable[str]] = None) -> int:
    """Main CLI entry point."""
    parser = _build_parser()
    if args is None:
        args = sys.argv[1:]

    parsed_args, remainder = parser.parse_known_args(list(args))
    exit_code = _dispatch(parser, parsed_args, remainder)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
