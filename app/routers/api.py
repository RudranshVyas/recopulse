from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import db, ml

router = APIRouter(prefix="/api", tags=["api"])


class SessionFeatures(BaseModel):
    """Session-level inputs. Every field is optional — omitted fields fall back to
    the population defaults in app/ml.py. Nothing here derives from the purchases
    table, which is a perfect leak (see data_profile.md)."""
    n_view: int = Field(3, ge=0)
    n_click: int = Field(1, ge=0)
    n_add_to_cart: int = Field(0, ge=0)
    n_remove_from_cart: int = Field(0, ge=0)
    n_add_to_wishlist: int = Field(0, ge=0)
    n_remove_from_wishlist: int = Field(0, ge=0)
    n_products: int = Field(3, ge=0)
    mean_dwell_ms: int = Field(12000, ge=0)
    max_dwell_ms: int = Field(30000, ge=0)
    duration_s: int = Field(300, ge=0)
    hour: int = Field(14, ge=0, le=23)
    day_of_week: int = Field(2, ge=0, le=6)
    prior_sessions: int = Field(0, ge=0)
    tenure_days: int = Field(180, ge=0)
    age: int = Field(35, ge=0)
    device_type: str = "desktop"
    referrer_source: str = "organic_search"
    loyalty_tier: str = "bronze"
    income_level: str = "medium"


@router.get("/metrics/overview")
def overview():
    return db.one("SELECT * FROM overview")


@router.get("/products")
def products(category: str | None = None, limit: int = Query(20, ge=1, le=200)):
    sql = ("SELECT product_id, product_name, category, subcategory, brand, price, "
           "rating_avg, review_count, n_view, n_purchases, revenue, view_to_purchase "
           "FROM product_stats")
    params = []
    if category:
        sql += " WHERE category = ?"
        params.append(category)
    sql += " ORDER BY revenue DESC, n_view DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, params)
    if category and not rows:
        raise HTTPException(404, f"No products in category '{category}'")
    return {"count": len(rows), "category": category, "products": rows}


@router.get("/recommendations/{user_id}")
def recommendations(user_id: str, n: int = Query(5, ge=1, le=20)):
    if not db.one("SELECT 1 AS x FROM users WHERE user_id = ?", (user_id,)):
        raise HTTPException(404, f"Unknown user_id '{user_id}'")
    recs = ml.recommend(user_id, n=n)
    return {
        "user_id": user_id,
        "strategy": "popularity_fallback" if recs and recs[0]["cold_start"]
                    else "item_based_cf",
        "count": len(recs),
        "recommendations": recs,
    }


@router.post("/predict/conversion")
def predict_conversion(features: SessionFeatures):
    prob, factors = ml.predict_conversion(features.model_dump())
    return {
        "conversion_probability": round(prob, 4),
        "predicted_label": int(prob >= 0.5),
        "base_rate": round(ml.CONV_METRICS["base_rate"], 4),
        "lift_vs_base": round(prob / ml.CONV_METRICS["base_rate"], 2),
        "top_factors": factors,
        "model": ml.CONV_METRICS["best_model"],
    }


@router.get("/segments/{user_id}")
def segment(user_id: str):
    s = ml.segment_of(user_id)
    if not s:
        raise HTTPException(404, f"Unknown user_id '{user_id}'")
    return s


@router.get("/insights")
def insights():
    """Auto-derived commercial findings, computed from the precomputed aggregates."""
    kpi = db.one("SELECT * FROM overview")
    best = db.one("SELECT category, revenue FROM category_stats ORDER BY revenue DESC LIMIT 1")
    worst = db.one("SELECT category, view_to_purchase FROM category_stats "
                   "WHERE view_to_purchase IS NOT NULL ORDER BY view_to_purchase ASC LIMIT 1")
    leaky = db.query("SELECT product_name, n_view, n_purchases, view_to_purchase "
                     "FROM product_stats WHERE n_view >= 100 AND view_to_purchase IS NOT NULL "
                     "ORDER BY view_to_purchase ASC LIMIT 3")
    browsers = next((s for s in ml.SEG_METRICS["segments"].values()
                     if s["name"] == "Heavy Browsers"), None)

    out = [
        {"kind": "conversion",
         "headline": f"Overall conversion sits at {kpi['conversion_rate']:.2%}",
         "detail": f"{kpi['total_purchases']:,} purchases across {kpi['total_sessions']:,} "
                   f"sessions, generating ${kpi['total_revenue']:,.0f}."},
        {"kind": "category",
         "headline": f"{best['category']} leads revenue at ${best['revenue']:,.0f}",
         "detail": f"Lowest view-to-purchase category is {worst['category']} at "
                   f"{worst['view_to_purchase']:.3%}."},
        {"kind": "funnel_leak",
         "headline": "High-traffic products converting near zero",
         "detail": "; ".join(f"{p['product_name']} ({p['n_view']:,} views → "
                             f"{p['n_purchases']} purchases)" for p in leaky)},
        {"kind": "model",
         "headline": f"Cart activity drives conversion prediction "
                     f"(ROC-AUC {ml.CONV_METRICS['roc_auc']:.3f})",
         "detail": f"Removing cart features drops F1 from {ml.CONV_METRICS['f1']:.3f} to "
                   f"{ml.CONV_METRICS['ablation']['f1']:.3f} — the signal is concentrated "
                   f"in add-to-cart behaviour."},
    ]
    if browsers:
        out.append({
            "kind": "segment",
            "headline": f"{browsers['size']:,} users are Heavy Browsers "
                        f"({browsers['share']:.1%} of base)",
            "detail": f"They average {browsers['profile']['n_views']:.1f} views but only "
                      f"{browsers['profile']['n_purchases']:.2f} purchases — the largest "
                      f"untapped conversion pool."})
    return {"count": len(out), "insights": out}
