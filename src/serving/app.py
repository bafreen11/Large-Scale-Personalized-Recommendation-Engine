"""Step 7: recommendation API with a live A/B test of the two rankers.

Run from the project root:
    uvicorn src.serving.app:app --port 8080
    (optional) docker compose up -d redis triton  and  REDIS_URL=redis://localhost:6379/0 TRITON_URL=localhost:8000

    GET  /recommend/{user_id}?k=10&liked=1,260,1196   liked = movieIds, only used for brand-new users
    POST /feedback   {"user_id": 1, "movie_id": 260, "variant": "dcn"}   -> a click on a shown movie
    GET  /ab/report  CTR per variant + two-proportion z-test
Impressions and clicks are appended to data/ab_events.jsonl.
"""
import json
import time
from collections import Counter

from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.common.config import ROOT
from src.evaluation.metrics import two_proportion_ztest
from src.serving.recommender import VARIANTS, Recommender, ab_variant

EVENTS = ROOT / "data" / "ab_events.jsonl"
app = FastAPI(title="Personalized Recommendation Engine")
state = {}
impressions, clicks = Counter(), Counter()


@app.on_event("startup")
def load():
    state["rec"] = Recommender()


def log(event):
    with EVENTS.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


@app.get("/health")
def health():
    rec = state["rec"]
    return {"status": "ok", "cache": rec.cache.backend, "user_tower": rec.user_tower.backend,
            "rankers": {v: m.backend for v, m in rec.rankers.items()}}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100), liked: str = ""):
    variant = ab_variant(user_id)
    liked_ids = [int(x) for x in liked.split(",") if x.strip()]
    out = state["rec"].recommend(user_id, k, variant, liked_ids)
    impressions[variant] += len(out["items"])
    log({"type": "impression", "user_id": user_id, "variant": variant, "movie_ids": [i["movieId"] for i in out["items"]]})
    return {"user_id": user_id, "variant": variant, **out}


class Feedback(BaseModel):
    user_id: int
    movie_id: int
    variant: str


@app.post("/feedback")
def feedback(fb: Feedback):
    clicks[fb.variant] += 1
    log({"type": "click", **fb.model_dump()})
    return {"ok": True}


@app.get("/ab/report")
def ab_report():
    a, b = VARIANTS
    report = {v: {"impressions": impressions[v], "clicks": clicks[v],
                  "ctr": clicks[v] / impressions[v] if impressions[v] else None} for v in VARIANTS}
    if impressions[a] and impressions[b]:
        z, p = two_proportion_ztest(clicks[a], impressions[a], clicks[b], impressions[b])
        report["ztest"] = {"z": z, "p_value": p, "significant_at_0.05": p < 0.05}
    return report
