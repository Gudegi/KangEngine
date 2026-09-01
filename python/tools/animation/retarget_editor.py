from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--mode", choices=("angle", "ik"), required=True)
    parser.add_argument("-h", "--help", action="store_true")
    args, editor_args = parser.parse_known_args()

    if args.mode == "angle":
        from angle_retarget_editor import main as editor_main

        program = "angle_retarget_editor.py"
    else:
        from ik_retarget_editor import main as editor_main

        program = "ik_retarget_editor.py"
    if args.help:
        editor_args.append("--help")
    sys.argv = [program, *editor_args]
    editor_main()


if __name__ == "__main__":
    main()
