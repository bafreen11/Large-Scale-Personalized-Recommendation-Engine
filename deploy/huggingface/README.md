---
title: Movie Recommender API
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Two-tower + FAISS + DCN movie recommendations (MovieLens 25M)
---

# Movie Recommender API

Live API of the [Large-Scale Personalized Recommendation Engine](https://github.com/bafreen11/Large-Scale-Personalized-Recommendation-Engine):
two-tower retrieval (TensorFlow Recommenders) → FAISS → DCN-v2 / Wide & Deep ranking, with popularity and
content-based fallbacks for new users. Trained on an 8% user sample of MovieLens 25M.

| Endpoint | Example |
|---|---|
| `GET /recommend/{user_id}?k=10` | `/recommend/39?k=5`, a known user, ranked by DCN or Wide & Deep (A/B split) |
| `GET /recommend/{user_id}?liked=260,2571` | a new user who likes Star Wars and The Matrix, content-based |
| `GET /movies/search?q=matrix` | find movieIds |
| `GET /health` | backends in use |
| `GET /docs` | interactive documentation |

This Space runs the models in-process with an in-memory cache. The full setup (Redis + Triton + Airflow) is in the GitHub repository.
