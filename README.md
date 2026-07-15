---
title: RecoPulse
emoji: 🛒
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# RecoPulse

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/RudranshVyas/recopulse)

ML-powered e-commerce product intelligence. One FastAPI service serves both a
server-rendered dashboard (Jinja2 + HTMX + Plotly) and a JSON API over a real
multi-table dataset of **133,305 rows**.

Three models ship: item-based collaborative filtering, session conversion
prediction, and behavioral customer segmentation. Every page renders from real
data; every metric below is measured, not asserted.

**Read [`data_profile.md`](data_profile.md) first** — it documents the real schema,
the join graph, and a signal audit that materially changed what got built.

---

## Quickstart

```bash
pip install -r requirements.txt

python scripts/profile_data.py        # schema + signal audit (prints; backs data_profile.md)
python scripts/load_data.py           # CSVs -> cleaned SQLite + precomputed aggregates
python scripts/train_recommender.py   # item-item CF  -> models/recommender.pkl
python scripts/train_conversion.py    # RF + ablation  -> models/conversion.pkl
python scripts/train_segmentation.py  # K-Means        -> models/segmentation.pkl

uvicorn app.main:app --reload --port 7860
```

Open <http://127.0.0.1:7860>. API docs at `/docs`.

The DB and pickles are **not committed** (see `.gitignore`) — they are rebuilt by the
four scripts above, and by the Dockerfile at image build time. The raw CSVs *are*
committed (largest is 19MB, under GitHub's 50MB limit), so everything is reproducible
from source.

---

## Pages

| # | Route | Title in the UI | What it shows |
|---|---|---|---|
| 01 | `/` | Store overview | KPI cards, monthly session/conversion trends, revenue + view mix by category, category table |
| 02 | `/products` | Product performance | Viewed-vs-purchased, view→purchase by category, funnel scatter, high-traffic/low-conversion, low-rated/high-traffic, HTMX catalog explorer |
| 03 | `/recommend` | Product suggestions | Enter a `user_id` → top-5 recs with reason strings (HTMX fragment); cold-start path |
| 04 | `/conversion` | Purchase predictor | Compose a session → live conversion probability + top factors (HTMX fragment) |
| 05 | `/segments` | Shopper groups | Cluster distribution, behavioral radar, per-segment profiles, why k=3 |
| 06 | `/model` | How accurate is it? | Honest metrics: cart ablation, confusion matrix, ROC, feature importance, k-sweep, recommender coverage |

**UI copy is written for a non-technical reader.** No page mentions the stack, table
names, raw column names, or repo files. Model internals are translated at the edge:
`app/ml.py: FEATURE_LABELS` / `humanize()` maps feature names to plain English for both
the prediction factors and the importance chart, and chart axes are labelled in prose
("share of real buyers caught", not "true positive rate"). The honest limitations still
appear in full on `/model` — restated in plain language rather than removed.

## API

| Method | Route |
|---|---|
| GET | `/api/metrics/overview` |
| GET | `/api/products?category=&limit=` |
| GET | `/api/recommendations/{user_id}?n=` |
| POST | `/api/predict/conversion` |
| GET | `/api/segments/{user_id}` |
| GET | `/api/insights` |
| GET | `/health` |

```bash
curl localhost:7860/health
curl "localhost:7860/api/products?category=Electronics&limit=3"
curl localhost:7860/api/recommendations/0000780a-2126-4e84-9622-42ce0ea9b17a
curl -X POST localhost:7860/api/predict/conversion \
  -H 'Content-Type: application/json' \
  -d '{"n_view":8,"n_click":4,"n_add_to_cart":3,"mean_dwell_ms":25000}'
```

---

## Data pipeline & cleaning

`scripts/load_data.py` reads all six CSVs, cleans, joins, and writes SQLite plus
**precomputed aggregate tables** (`overview`, `category_stats`, `product_stats`,
`monthly_trend`, `session_features`) so no request ever scans the raw tables.

Cleaning decisions, all driven by what the audit actually found:

1. **Timestamps** carry sub-microsecond noise (`2023-01-16 12:57:18.000000115`).
   Parsed with `format="mixed"` and floored to the second.
2. **`products.rating_avg` is NULL iff `review_count == 0`** — verified structural, not
   missing data. Kept NULL, **never imputed to 0**; imputing would drag the "avg rating"
   KPI toward zero across the 548 unrated products. Aggregates use skipna.
3. **`reviews.purchase_id`** is null for 200 rows (16%) — unverified reviews. Kept, and
   surfaced as an `is_verified` flag rather than dropped.
4. **Categorical text** stripped and normalized across users/products/sessions/interactions.
5. **`product_description`** (long template text with unfilled tokens like `{material}`)
   is dropped; a 180-char `blurb` is kept. `review_text` truncated to 400 chars.
6. **Count columns** cast to int after zero-fill so the API returns `746`, not `746.0`.
7. **Indexes** added on every FK used at serving time.

Referential integrity was verified clean: **zero orphans on all 10 foreign keys**.
Row counts are asserted against the source CSVs on every run and the script exits
non-zero on mismatch.

---

## Models

### 1. Recommender — item-based CF

Implicit weights mapped to the **real** interaction vocabulary. The spec's assumed
`purchase` interaction type **does not exist** in this data, so purchase signal is
joined from the `purchases` table instead:

| signal | weight |
|---|---|
| `view` | 1 |
| `click` | 2 |
| `add_to_wishlist` | 3 |
| `add_to_cart` | 4 |
| purchase *(from `purchases`)* | 5 |
| `remove_from_cart` | −2 |
| `remove_from_wishlist` | −1 |

Net-negative user-item pairs clip to 0 — in implicit feedback, disinterest is absence,
not a negative rating. Cosine similarity over L2-normalized item columns of a
6,944 × 1,000 sparse matrix, pruned to the top 50 neighbours per item.

| metric | value |
|---|---|
| users with CF history | 6,944 / 10,000 (69.4%) |
| matrix density | 0.77% |
| similarity edges | 45,824 |
| catalog coverage | 392 / 1,000 distinct items across a 500-user sample (39.2%) |
| cold-start users served | 3,056 (31%) via popularity fallback |

### 2. Conversion — RandomForest

Session-level binary target `is_converted`, base rate **7.46%**, stratified 75/25 split.

| model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| LogisticRegression (baseline) | 0.9565 | 0.5793 | 0.7917 | 0.6690 |
| **RandomForest (ships)** | **0.9590** | **0.8145** | **0.7806** | **0.7972** |
| LightGBM (tested, rejected) | 0.9574 | — | — | 0.7784 |

**LightGBM was tested and lost** on both ROC-AUC and F1, so per spec the stack stays
sklearn-only and LightGBM is not a dependency.

Confusion matrix (4,829 held-out sessions):

|  | pred: no | pred: yes |
|---|---|---|
| **actual: no** | 4,405 | 64 |
| **actual: yes** | 79 | 281 |

### 3. Segmentation — K-Means, k=3

12 standardized behavioral features across **all 10,000 users** (not just the 6,944
active ones — dormant signups are a real commercial segment).

Silhouette peaks at **k=2 (0.564)**, but that split is degenerate: it only separates
"engaged" from "not engaged", which needs no model, and it buries the dormant cohort.
Excluding the trivial split, **k=3 (silhouette 0.498)** wins and sits at the inertia
elbow. Both numbers are shown on `/model` — the rejected one included.

| segment | users | share | avg views | avg purchases | AOV |
|---|---|---|---|---|---|
| Low-Touch Visitors | 6,674 | 66.7% | 1.9 | 0.00 | $0.08 |
| **Heavy Browsers** | 2,292 | 22.9% | 12.4 | 0.10 | $3.94 |
| High-Value Buyers | 1,034 | 10.3% | 9.0 | 1.43 | $103.88 |

Segment names are derived from cluster profiles relative to the population, not
hand-assigned.

---

## ⚠️ Honest limitations

This dataset is synthetic. Several results look better than they are, and the app is
built to say so rather than hide it.

**1. The conversion model is largely learning one generator rule.**
No session without an add-to-cart event ever converted — **0 out of 10,944**. That is
deterministic, not a subtle pattern. Cart features are kept (they causally precede
purchase and are observable mid-session, so a real-time model would legitimately have
them), but they dominate. Retraining with cart features removed:

| | with cart | without cart |
|---|---|---|
| ROC-AUC | 0.959 | 0.869 |
| F1 | 0.797 | **0.371** |
| Recall | 0.781 | **0.286** |

**The 0.959 ROC-AUC is real but should not be read as model quality.** It mostly
reflects how the data was generated. This ablation is shown at the top of `/model`.

**2. `purchases` is a perfect leak and is fully excluded.**
`set(purchases.session_id) == set(converted sessions)`, exactly. Any purchase-derived
feature predicts the target with 100% accuracy and is unavailable at prediction time.
Nothing from that table reaches the conversion model.

**3. Device, referrer, and dwell time carry no signal.**
Measured conversion rates: device 7.26–7.53%, referrer 7.24–8.05%, and mean dwell is
slightly *lower* for converted sessions (18,060ms vs 19,617ms). The generator did not
condition on them. They are kept in the feature set and their near-zero importances
(~0.003) are displayed on `/model` — an honest negative result, not a bug.

**4. Recommender coverage is limited.**
At 0.77% density, item-item similarity for long-tail products rests on few
co-occurrences, so tail recommendations are lower-confidence than their scores suggest.
Only 39.2% of the catalog surfaces in top-10 recs, 33 products have zero interactions,
and 31% of users never reach the CF path at all. There is **no offline
precision@k / recall@k evaluation** — no held-out temporal split was built for the
recommender, so its quality is reported as coverage only, not accuracy.

**5. Segmentation is descriptive, not validated.**
Silhouette 0.498 indicates moderate, not crisp, separation. The segments are a useful
lens, not ground truth.

**6. Reviews sit on an inconsistent timeline.**
Reviews start 2022-01-13, a full year before the first session (2023-01-04), and 200
have no `purchase_id`. Reviews cannot be causally chained to sessions, so they are used
only for product rating aggregates and segmentation counts — never for conversion features.

---

## Deploy

The image is host-agnostic. `CMD` honours an injected `$PORT` and falls back to 7860,
so the same container runs on Cloud Run, Render, or HF Spaces with no changes.

Measured on the built image, at each host's free-plan limits:

| | Render free (0.1 CPU) | Cloud Run (1 CPU) |
|---|---|---|
| memory in use, after serving every page | **177 MiB** / 512MB | **186 MiB** / 512Mi |
| cold start → first healthy response | **~26 s** | **~3.5 s** |
| page latency once warm | 0.14 – 0.40 s | — |
| OOM under the cap | no | no |

Image is 949 MB; the container writes nothing at runtime and serves read-only.

### Primary: Render (free, no credit card)

`render.yaml` is committed, so this is a Blueprint deploy. One click:

**[→ Deploy this repo to Render](https://dashboard.render.com/blueprint/new?repo=https://github.com/RudranshVyas/recopulse)**

Or manually: Render dashboard → **New → Blueprint** → select the repo.

Either way Render reads `render.yaml` (`runtime: docker`, `plan: free`), authorises access
to the repo, and builds the Dockerfile. The build creates the SQLite DB and trains all
three models, so the image ships with them baked in. First build takes a few minutes
(pip install plus training); it lands at `https://recopulse.onrender.com`.

Render injects `$PORT` and the container reads it — no configuration needed. The health
check is wired to `/health`.

**Free-plan behaviour, measured — not guessed.** The instance is 512MB / **0.1 CPU**, and
that CPU limit is the thing that shows. Running the real image at `--cpus=0.1
--memory=512m`:

| | |
|---|---|
| cold start → first healthy response | **~26 s** |
| memory in use | **177 MiB** of 512MB, no OOM |
| page latency once warm | **0.14 – 0.40 s** |

The slow part is one-time Python import and model loading at a tenth of a CPU. Once warm,
pages are fast, because every dashboard figure is read from a precomputed aggregate table
rather than computed per request.

Also: free services **sleep after 15 minutes idle**, so the first visitor after a quiet
spell waits for that cold start plus Render's own spin-up — call it a minute. Budget is
750 instance-hours/month.

No credit card required.

### Alternative: Google Cloud Run (free, permanent — if billing will set up)

Better than Render on every axis except signup: scales to zero, and the free tier **resets
monthly and never expires** (2M requests, 180k vCPU-seconds, 360k GiB-seconds). At 1 CPU
the same image cold-starts in **~3.5 s** rather than 26 s.

The catch is that it **requires an attached billing account**, and that step is not always
passable — setup can fail with errors like `OR_BACR2_44` even after a successful payment
method, which is a Google-side verification issue with no fix in this repo.

```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy recopulse \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 2 \
  --cpu-boost
```

Cloud Build builds the Dockerfile (training included), pushes to Artifact Registry, and
returns an HTTPS URL. No TLS setup, no VM.

- `--max-instances 2` caps the blast radius. Since billing is attached, bound the spend
  rather than trusting the free tier to hold — add a budget alert at $1 in
  **Billing → Budgets & alerts**.
- `--cpu-boost` shortens the cold start; `--min-instances 0` (the default) keeps it free
  while idle.
- Cloud Run injects `PORT=8080`. The container reads it. Do not pass `--port`.
- `.gcloudignore` keeps `models/` and the built DB out of the upload — they are rebuilt
  inside the image. `data/raw/*.csv` **is** uploaded; the build needs it.

### Hugging Face Spaces — check before relying on it

The YAML frontmatter at the top of this README is retained, so this repo still deploys as
a Space (`sdk: docker`, `app_port: 7860`) if you have PRO.

**As of July 2026 this may not work on a free account.** Free accounts report being unable
to select CPU Basic at Space creation, with Docker marked "Paid" and only ZeroGPU offered
— affecting existing Spaces too. The pricing page still advertises CPU Basic as free, so
the two contradict each other and HF has not commented. Verify before depending on it;
PRO is $9/month.

### Locally

```bash
docker build -t recopulse .
docker run -p 7860:7860 recopulse            # http://localhost:7860
docker run -e PORT=8080 -p 8080:8080 recopulse   # how Cloud Run/Render invoke it
```

---

## Structure

```
app/
  main.py            FastAPI app, /health, mounts
  db.py              SQLite connection + query helpers
  ml.py              loads the 3 pickles once at import; predict/recommend/segment
  charts.py          Plotly figures themed to the design system
  routers/
    pages.py         HTML pages + HTMX fragments
    api.py           JSON API
  templates/         Jinja2 (base + 6 pages + partials/)
  static/app.css     design system
scripts/
  profile_data.py    schema profile + signal audit
  load_data.py       CSV -> SQLite + precomputed aggregates
  train_recommender.py
  train_conversion.py
  train_segmentation.py
data/raw/*.csv       source data (committed)
models/              pickles + metrics json (built, gitignored)
data_profile.md      real schema, join graph, signal audit
Dockerfile           builds DB + trains models at image build; honours $PORT
render.yaml          Render Blueprint (free plan, docker runtime)
.gcloudignore        what Cloud Build uploads
requirements.txt
```

## Stack

FastAPI · Jinja2 · HTMX · Plotly (server-side) · Tailwind (CDN) · scikit-learn ·
pandas / numpy / scipy (sparse) · SQLite · Docker

**Design.** Deliberately not a default-Tailwind dashboard: a Swiss-editorial /
financial-terminal system — hairline rules instead of shadowed rounded cards,
Instrument Serif numerals against IBM Plex Mono labels, acid-lime signal on near-black
ink, tabular figures, a grain overlay, and one orchestrated staggered page load.

## Choices made where the spec left a gap

- **Charts:** Plotly rendered server-side to HTML strings and embedded; `plotly.js` from
  CDN handles only hydration.
- **Conversion demo defaults:** omitted API fields fall back to population defaults
  (`app/ml.py: DEFAULTS`) rather than erroring.
- **`/api/insights`:** derived from precomputed aggregates + model metrics at request
  time; no LLM, no hand-written copy.
- **Segmentation scope:** all 10,000 users, including the 3,056 with zero history.
- **Similarity pruning:** top-50 neighbours per item — bounds memory and cuts tail noise.
- **Top factors:** global model importance shown alongside the value *this* session
  supplied, rather than per-prediction SHAP (kept dependency-light).
