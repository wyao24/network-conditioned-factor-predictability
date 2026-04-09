from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\wyao2\Documents\MATH279Project")
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_outputs"
PRIORITY_ROOT = ANALYSIS_ROOT / "priority4_6"
ALPHAMARK_ROOT = PROJECT_ROOT / "_external" / "alphamark"
ALPHAMARK_FEATURES_DIR = ANALYSIS_ROOT / "alphamark_input" / "daily_features_pkl"
ALPHAMARK_OUTPUT_ROOT = ANALYSIS_ROOT / "alphamark_output"
PYTHON_EXE = Path(r"C:\Users\wyao2\anaconda3\python.exe")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaMark on exported project signals.")
    parser.add_argument("--features-dir", default=str(ALPHAMARK_FEATURES_DIR))
    parser.add_argument("--output-root", default=str(ALPHAMARK_OUTPUT_ROOT))
    parser.add_argument("--interval-start", default="2003-01-01")
    parser.add_argument("--interval-end", default="2023-12-31")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_root = Path(args.output_root)
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

    cmd = [str(PYTHON_EXE), str(ALPHAMARK_ROOT / "main.py")]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ALPHAMARK_ROOT), env=env)


if __name__ == "__main__":
    main()
