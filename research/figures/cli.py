"""Shared argument parsing for the figure scripts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .data import Results
from .style import DEFAULT_FORMATS, use_paper_style


def build_parser(description: str, epilog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=Path("research/outputs"),
        help="Directory holding the result artefacts (see figures/data.py).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("research/outputs/figures"),
        help="Where the rendered figures are written.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Render from synthetic data seeded by the published summary "
             "statistics, to review layout before the real artefacts exist. "
             "Output is stamped and suffixed _DEMO.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=list(DEFAULT_FORMATS),
        choices=["pdf", "tiff", "tif", "png", "eps", "svg"],
        help="Output formats (default: PDF for submission, TIFF for production, PNG for drafts).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def setup(args: argparse.Namespace) -> Results:
    """Configure logging and style, and return the results bundle."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    use_paper_style()
    if args.demo:
        logging.getLogger(__name__).warning(
            "DEMO MODE: figures are rendered from synthetic data and stamped "
            "accordingly -- do not use them as results"
        )
    return Results(root=args.results, demo=args.demo)
