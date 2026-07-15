from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import api, pages

app = FastAPI(
    title="RecoPulse",
    description="ML-powered e-commerce product intelligence. Recommendations, "
                "conversion prediction, and behavioral segmentation over a "
                "133,305-row multi-table dataset.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
          name="static")
app.include_router(api.router)
app.include_router(pages.router)


@app.get("/health", tags=["ops"])
def health():
    from app import ml
    return {"status": "ok", "models": ["recommender", "conversion", "segmentation"],
            "conversion_auc": round(ml.CONV_METRICS["roc_auc"], 4)}
