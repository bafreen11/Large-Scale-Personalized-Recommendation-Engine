"""Step 7: recommendation API with a live A/B test of the two rankers.

Run from the project root:
    uvicorn src.serving.app:app --port 8080
    (optional) docker compose up -d redis triton  and  REDIS_URL=redis://localhost:6379/0 TRITON_URL=localhost:8000

    GET  /recommend/{user_id}?k=10&liked=1,260,1196   liked = movieIds, only used for brand-new users
    POST /feedback   {"user_id": 1, "movie_id": 260, "variant": "dcn"}   -> a click on a shown movie
    GET  /ab/report  CTR per variant + two-proportion z-test
    GET  /movies/search?q=matrix   movieIds for the "movies you like" picker of the web page
Impressions and clicks are appended to data/ab_events.jsonl.
"""
import json
import os
import time
from collections import Counter

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.common.config import ROOT
from src.evaluation.metrics import two_proportion_ztest
from src.serving.recommender import VARIANTS, Recommender, ab_variant

EVENTS = ROOT / "data" / "ab_events.jsonl"
app = FastAPI(title="Personalized Recommendation Engine")
# the web/ page (Vercel / Netlify) calls the API from another origin; CORS_ORIGINS restricts it
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
state = {}
impressions, clicks = Counter(), Counter()


@app.on_event("startup")
def load():
    state["rec"] = Recommender()
    state["titles_lower"] = np.char.lower(state["rec"].cat.titles.astype(str))


def log(event):
    with EVENTS.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/movies/search")
def search_movies(q: str = Query(min_length=2, max_length=100), limit: int = Query(10, ge=1, le=50)):
    """Title search, most-rated first."""
    cat = state["rec"].cat
    hits = np.flatnonzero(np.char.find(state["titles_lower"], q.lower()) >= 0)
    hits = hits[np.argsort(-cat.pop_all[hits], kind="stable")][:limit]
    return [{"movieId": int(cat.movie_ids[i]), "title": str(cat.titles[i])} for i in hits]


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
