"""Stage the serving-only models and data that the Render image is built from.

    python scripts/stage_render_artifacts.py

Render builds from the GitHub repository, not from a folder we upload, so the slim artifacts
(~90 MB: user tower, both rankers, the FAISS index, items/users and a trimmed train.parquet)
have to be committed. They live in deploy/render/artifacts/ and are copied into the image by
deploy/render/Dockerfile. Re-run this after retraining, then commit the result.

The full data/ and models/ folders (~5 GB) stay out of git; only this slice is tracked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_hf_space import ROOT, copy_artifacts, report  # noqa: E402

OUT = ROOT / "deploy" / "render" / "artifacts"

if __name__ == "__main__":
    seen_rows = copy_artifacts(OUT, with_src=False)
    report(OUT, seen_rows)
