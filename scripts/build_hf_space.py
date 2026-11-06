"""Assemble (and optionally upload) the Hugging Face Space that serves the recommendation API.

    python scripts/build_hf_space.py                 # build/hf_space/ only; test with docker build
    python scripts/build_hf_space.py --upload        # + create/update <your-hf-user>/movie-recommender
                                                     #   (log in first: hf auth login)

NOTE (Nov 2026): Hugging Face now requires a PRO subscription to run Docker or Gradio Spaces
on free cpu-basic hardware -- create_repo returns 402 Payment Required. Only static Spaces are
free. The live API is deployed on Render instead; see render.yaml and deploy/render/Dockerfile.
This script is kept because the bundle it builds is the same slice of models and data.

The Space gets only what serving needs (~100 MB instead of the 5 GB data folder):
src/, the user tower, both rankers, the FAISS index, items + users snapshots, and train.parquet
cut down to (userId, movieId), used to drop already-seen movies.
"""
import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf_space"
SPACE = "movie-recommender"
NOISE = shutil.ignore_patterns("__pycache__", "*.crc", "_SUCCESS")


def copy_artifacts(out, with_src=True):
    """Copy the serving-only slice of src/, models/ and data/training into `out`.

    Shared with scripts/stage_render_artifacts.py. Returns the number of seen (userId, movieId)
    pairs. with_src=False when the host builds src/ straight from the repository (Render).
    """
    if out.exists():
        shutil.rmtree(out)
    if with_src:
        shutil.copytree(ROOT / "src", out / "src", ignore=NOISE)
    for d in ["two_tower/user_tower", "ranker_dcn", "ranker_wide_deep"]:
        shutil.copytree(ROOT / "models" / d, out / "models" / d, ignore=NOISE)
    (out / "models" / "faiss").mkdir(parents=True)
    shutil.copy2(ROOT / "models/faiss/items_ivf.faiss", out / "models/faiss/")

    training = out / "data" / "training"
    training.mkdir(parents=True)
    for name in ["items", "users"]:
        pq.write_table(pq.read_table(ROOT / f"data/training/{name}.parquet"), training / f"{name}.parquet")
    seen = pq.read_table(ROOT / "data/training/train.parquet", columns=["userId", "movieId"])
    pq.write_table(seen, training / "train.parquet")
    return seen.num_rows


def report(out, seen_rows):
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"built {out}  ({size:.0f} MB, {seen_rows:,} seen pairs)")


def build():
    seen_rows = copy_artifacts(OUT)
    for f in (ROOT / "deploy" / "huggingface").iterdir():
        shutil.copy2(f, OUT / f.name)
    report(OUT, seen_rows)


def upload():
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{SPACE}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=repo_id, repo_type="space",
                      commit_message="Deploy recommendation API")
    user, name = repo_id.split("/")
    print(f"Space:   https://huggingface.co/spaces/{repo_id}")
    print(f"API URL: https://{user.lower()}-{name}.hf.space   (live once the Space has finished building)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="upload to Hugging Face after building")
    args = ap.parse_args()
    build()
    if args.upload:
        upload()
