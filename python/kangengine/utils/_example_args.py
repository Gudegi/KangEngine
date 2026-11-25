"""Shared renderer selection for standalone examples."""

import argparse


def render_example_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--renderer", choices=("native", "ovrtx"), default="native",
        help="Scene renderer; individual examples may require Native-only features.",
    )
    return parser
