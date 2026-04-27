from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmhe_v1.video_flow import run_video_flow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the single-video action recognition flow.")
    parser.add_argument(
        "--config",
        default="configs/experiments/v1_baseline.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/video_flow",
        help="Directory for video_result.json and video_summary.md.",
    )
    args = parser.parse_args()

    paths = run_video_flow(
        config_path=args.config,
        video_path=args.video,
        output_dir=args.output_dir,
    )
    print(f"result_json={paths['result_json_path']}")
    print(f"summary={paths['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
