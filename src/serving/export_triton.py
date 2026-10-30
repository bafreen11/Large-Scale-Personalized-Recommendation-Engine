"""Step 7: copy the trained SavedModels into Triton's model repository.

Run from the project root:
    python -m src.serving.export_triton
    docker compose up -d triton
    TRITON_URL=localhost:8000 uvicorn src.serving.app:app --port 8080    (serving now calls Triton)

Layout (config.pbtxt files are in git, model weights are not):
    triton/model_repository/<name>/config.pbtxt
    triton/model_repository/<name>/1/model.savedmodel/
"""
import shutil

from src.common.config import MODELS_DIR, ROOT

REPO = ROOT / "triton" / "model_repository"
MODELS = {"user_tower": MODELS_DIR / "two_tower" / "user_tower",
          "ranker_dcn": MODELS_DIR / "ranker_dcn",
          "ranker_wide_deep": MODELS_DIR / "ranker_wide_deep"}


def main():
    for name, src in MODELS.items():
        dst = REPO / name / "1" / "model.savedmodel"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        print(f"{name:18s} -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
