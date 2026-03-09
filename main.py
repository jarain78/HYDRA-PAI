#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path


def list_apps(src_dir: Path) -> list[Path]:
    """List candidate Streamlit apps in src/ (top-level .py files)."""
    if not src_dir.is_dir():
        return []
    apps = sorted(
        [p for p in src_dir.glob("*.py") if p.is_file() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    )
    return apps


def pick_from_menu(apps: list[Path]) -> Path:
    """Interactive CLI menu to pick a Streamlit app."""
    print("\n=== HYDRA-PAI Launcher ===")
    print("Select the example you want to run (from ./src):\n")
    for i, p in enumerate(apps, start=1):
        print(f"  [{i}] {p.name}")
    print()

    while True:
        s = input(f"Enter a number (1-{len(apps)}) or 'q' to quit: ").strip().lower()
        if s in {"q", "quit", "exit"}:
            raise SystemExit(0)
        try:
            idx = int(s)
            if 1 <= idx <= len(apps):
                return apps[idx - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.\n")


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src/arms"

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "HYDRA-PAI Streamlit launcher.\n\n"
            "If --app is not provided, an interactive menu will be shown with the examples found in ./src."
        ),
    )
    parser.add_argument("--port", type=int, default=8501, help="Streamlit server port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    parser.add_argument(
        "--app",
        type=str,
        default="",
        help="Path to Streamlit app (.py). If relative, resolved from repo root.",
    )

    args, unknown = parser.parse_known_args()

    # Resolve app
    if args.app:
        app_path = Path(args.app)
        app_path = app_path if app_path.is_absolute() else (repo_root / app_path)
        app_path = app_path.resolve()
        if not app_path.is_file():
            raise FileNotFoundError(f"--app not found: {app_path}")
    else:
        apps = list_apps(src_dir)
        if not apps:
            raise FileNotFoundError(
                f"No .py examples found in: {src_dir}\n"
                "Either add examples to src/ or run: python main.py --app path/to/app.py"
            )
        app_path = pick_from_menu(apps)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", str(args.host),
    ] + unknown

    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())