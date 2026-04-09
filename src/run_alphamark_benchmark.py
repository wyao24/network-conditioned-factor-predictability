from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROJECT_ROOT = Path(r"C:\Users\wyao2\Documents\MATH279Project")
LEGACY_PYTHON_EXE = Path(r"C:\Users\wyao2\anaconda3\python.exe")


def build_plot_config() -> dict[str, object]:
    return {
        "H2_targets": ["fret_cc"],
        "H2_bets": ["betsize_unit"],
        "H3_targets": ["fret_cc"],
        "H3_bets": ["betsize_unit"],
        "variables_temporal_plot": ["pnl", "ppd", "nrTrades", "sizeNotional"],
        "bar_page_vars": ["signal", "bet_size_col"],
        "bar_x_vars": ["target"],
        "bar_metrics": ["pnl", "ppd", "sharpe", "hit_ratio", "sizeNotional", "n_trades", "market_corr"],
        "interval_start": "2003-01-01",
        "interval_end": "2023-12-31",
    }


def _resolve_path(path_str: str | None, default: Path) -> Path:
    if path_str:
        return Path(path_str).expanduser().resolve()
    return default.resolve()


def _default_python_exe() -> Path:
    if LEGACY_PYTHON_EXE.exists():
        return LEGACY_PYTHON_EXE.resolve()
    return Path(sys.executable).resolve()


def _require_existing_path(path: Path, parser: argparse.ArgumentParser, label: str) -> None:
    if not path.exists():
        parser.error(f"{label} does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaMark on exported project signals.")
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Project root path. Defaults to the repo root derived from this file "
            "(or legacy path if it exists)."
        ),
    )
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Analysis outputs root. Defaults to <project-root>/analysis_outputs.",
    )
    parser.add_argument(
        "--alphamark-root",
        default=None,
        help="AlphaMark repository root. Defaults to <project-root>/_external/alphamark.",
    )
    parser.add_argument(
        "--python-exe",
        default=None,
        help="Python executable used to run AlphaMark. Defaults to current interpreter.",
    )
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--interval-start", default="2003-01-01")
    parser.add_argument("--interval-end", default="2023-12-31")
    args = parser.parse_args()

    default_project_root = LEGACY_PROJECT_ROOT if LEGACY_PROJECT_ROOT.exists() else DEFAULT_PROJECT_ROOT
    project_root = _resolve_path(args.project_root, default_project_root)
    analysis_root = _resolve_path(args.analysis_root, project_root / "analysis_outputs")
    alphamark_root = _resolve_path(args.alphamark_root, project_root / "_external" / "alphamark")
    python_exe = _resolve_path(args.python_exe, _default_python_exe())
    features_dir = _resolve_path(args.features_dir, analysis_root / "alphamark_input" / "daily_features_pkl")
    output_root = _resolve_path(args.output_root, analysis_root / "alphamark_output")

    _require_existing_path(project_root, parser, "Project root")
    _require_existing_path(analysis_root, parser, "Analysis root")
    _require_existing_path(alphamark_root, parser, "AlphaMark root")
    _require_existing_path(features_dir, parser, "Features directory")
    _require_existing_path(python_exe, parser, "Python executable")

    alphamark_main = alphamark_root / "main.py"
    if not alphamark_main.exists():
        parser.error(f"AlphaMark entrypoint not found: {alphamark_main}")

    output_root.mkdir(parents=True, exist_ok=True)

    plot_cfg = build_plot_config()
    plot_cfg["interval_start"] = args.interval_start
    plot_cfg["interval_end"] = args.interval_end
    cfg_path = output_root / "alphamark_plot_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(plot_cfg, f, indent=2)

    env = os.environ.copy()
    env["FP_FEATURES_DIR"] = str(features_dir)
    env["FP_OUTPUT_ROOT"] = str(output_root)
    env["FP_INTERVAL_START"] = args.interval_start
    env["FP_INTERVAL_END"] = args.interval_end
    env["FP_PLOT_CONFIG"] = str(cfg_path)
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [str(python_exe), str(alphamark_main)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(alphamark_root), env=env)


if __name__ == "__main__":
    main()
