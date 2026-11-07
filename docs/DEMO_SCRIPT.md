# Demo recording script (about 7 minutes)

Shows the deployed system and the measured results. No Docker, no local servers: everything runs
from a browser against the live deployment, so there is very little that can break while recording.

Record with **Xbox Game Bar** (`Win + Alt + R` starts and stops, saved to `Videos\Captures`) or OBS
Studio. Use a 1080p screen, close notifications and chat apps, and hide the browser bookmarks bar
(`Ctrl + Shift + B`).

## Before recording (about 10 minutes, not recorded)

1. **Wake the API.** It sleeps after 15 idle minutes on the free plan and takes about a minute to
   come back. Open <https://recsys-api-bafreen11.onrender.com/health> and wait for the JSON.
2. **Warm the page with a user you will not demo.** Open the demo page and press Recommend for
   **user 3373**. This confirms everything works and leaves user 39 uncached, so your first
   on-camera request shows the real pipeline latency and the `freshly computed` badge.
3. **Open these tabs, in this order:**
   - <https://github.com/bafreen11/Large-Scale-Personalized-Recommendation-Engine>
   - <https://movie-recommender-bafreen11.netlify.app>
   - <https://recsys-api-bafreen11.onrender.com/docs>
   - `reports/ab_test_report.json` in VS Code
4. Zoom the browser to 125% (`Ctrl + +`) so text is readable in the recording.

## Recording

| Time | Show | Say |
|---|---|---|
| 0:00 | GitHub README, top of the page | "A movie recommender built on MovieLens 25M: 25 million ratings, 162,000 users, 62,000 movies. It runs on a laptop with four cores, 12 GB of RAM and no GPU, and it is deployed and live." |
| 0:30 | README architecture diagram | "Ratings go through a time-based split, Spark feature engineering, a two-tower retrieval model, a FAISS index for nearest-neighbour search, then a DCN or Wide & Deep ranker re-scores the candidates. New users fall back to popularity or content-based picks." |
| 1:15 | Demo page, **Existing user**, id **39**, k = 10, press Recommend | "This is the live system. One request: the user tower produces an embedding, FAISS retrieves candidates, the ranker re-scores them. About a second end to end, including the network." |
| 1:45 | Point at the badges, then the score bars | "The badge says which ranker served it — users are split 50/50 between DCN and Wide & Deep for a live A/B test. The bars are the ranker score on a zero-to-one scale, so you can see the model is confident but the top ten are close together." |
| 2:15 | Press Recommend again for the same user | "Same request again: `served from cache`, two milliseconds instead of a second. Lists are cached per user, variant and list length." |
| 2:45 | **New user** tab, pick *Star Wars* and *The Matrix*, Recommend | "A brand-new user has no history, so the two-tower model has nothing to embed. The cold-start path takes the movies they picked and finds similar ones by content — Empire Strikes Back, Return of the Jedi, Terminator. The rows say `no ranker score` because these come from the fallback, not the ranker." |
| 3:30 | API tab, `/docs`, expand `GET /health` → Try it out | "The same API, documented. Health shows which backends are in use: in-memory cache, local models. On the full local stack those become Redis and Triton." |
| 4:00 | `GET /movies/search` → `q=matrix` | "Title search, most-rated first — this is what the movie picker uses." |
| 4:30 | README results table | "Measured on 432 held-out users. The DCN ranker reaches 0.776 AUC. FAISS finds 90.8% of the exact top-200 in 0.105 milliseconds instead of 0.77 — about seven times faster. Serving is 7.9 milliseconds uncached." |
| 5:15 | README A/B test table | "The A/B test replays the test period with a stable hash split. DCN is ahead on click-through, 5.3% against 4.4%, but the z-test gives p = 0.24 — that is not significant at this sample size, and I report it that way rather than claiming a win." |
| 5:45 | README results table again, point at the popularity row | "One honest result: a 30-day popularity list beats the learned models on this 8% user sample. Popularity is a hard baseline on small samples, and I did not want to hide that. Training on more users is the main thing I would change." |
| 6:15 | `reports/ab_test_report.json` in VS Code | "Every number comes from a report file in the repository, not from a slide." |
| 6:45 | Back to the demo page | "Live demo and API links are at the top of the README. Thanks for watching." |

## If the API is asleep on camera

The page shows "Waking up the free server" after four seconds. Say: *"This is on a free tier that
sleeps when idle — it takes about a minute to wake."* Keep talking through the architecture while
it loads. This is why step 1 of the preparation exists.

## After recording

Trim the start and end in the Photos app (`Videos\Captures` → open → Edit → Trim) and upload.
