from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="MLLM hallucination evaluation framework entry point.")
    parser.add_argument("--version", action="store_true", help="Print framework version.")
    args = parser.parse_args()
    if args.version:
        print("hallucination-eval-framework 0.1.0")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
