from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates

from app import charts, db, ml

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def page(request, name, active, **ctx):
    return templates.TemplateResponse(request, name, {"active": active, **ctx})


@router.get("/")
def overview(request: Request):
    kpi = db.one("SELECT * FROM overview")
    trend = db.query("SELECT * FROM monthly_trend ORDER BY month")
    cats = db.query("SELECT * FROM category_stats ORDER BY revenue DESC")

    months = [t["month"][:7] for t in trend]
    figs = {
        "sessions": charts.render(
            charts.line(months, [t["sessions"] for t in trend], "store visits"), 260),
        "conversion": charts.render(
            charts.line(months, [round(t["conversion_rate"] * 100, 2) for t in trend],
                        "% of visits that buy", fill=False, color=charts.WARM), 260),
        "revenue": charts.render(
            charts.bar([c["category"] for c in cats], [round(c["revenue"], 2) for c in cats]), 300),
        "mix": charts.render(
            charts.donut([c["category"] for c in cats], [c["views"] for c in cats]), 300),
    }
    return page(request, "index.html", "/", kpi=kpi, figs=figs, cats=cats)


@router.get("/products")
def products(request: Request):
    cats = db.query("SELECT * FROM category_stats ORDER BY revenue DESC")
    cat_names = [c["category"] for c in cats]

    top_viewed = db.query(
        "SELECT product_name, category, n_view, n_purchases FROM product_stats "
        "ORDER BY n_view DESC LIMIT 12")
    leaky = db.query(
        "SELECT product_name, category, n_view, n_purchases, view_to_purchase "
        "FROM product_stats WHERE n_view >= 100 AND view_to_purchase IS NOT NULL "
        "ORDER BY view_to_purchase ASC LIMIT 10")
    lowrated = db.query(
        "SELECT product_name, category, rating_avg, review_count, n_view "
        "FROM product_stats WHERE rating_avg IS NOT NULL AND rating_avg <= 3 "
        "AND n_view >= 50 ORDER BY n_view DESC LIMIT 10")
    scat = db.query(
        "SELECT product_name, n_view, view_to_purchase, revenue FROM product_stats "
        "WHERE n_view >= 20 AND view_to_purchase IS NOT NULL")

    figs = {
        "viewed_vs_purchased": charts.render(charts.grouped_bar(
            [p["product_name"][:22] for p in top_viewed],
            {"times viewed": [p["n_view"] for p in top_viewed],
             "times bought (×20)": [p["n_purchases"] * 20 for p in top_viewed]}), 320),
        "conv_by_cat": charts.render(charts.bar(
            cat_names, [round((c["view_to_purchase"] or 0) * 100, 3) for c in cats],
            horizontal=True, color=charts.SIGNAL), 320),
        "revenue": charts.render(charts.bar(
            cat_names, [round(c["revenue"], 2) for c in cats], color=charts.WARM), 300),
        "scatter": charts.render(charts.scatter(
            [s["n_view"] for s in scat], [s["view_to_purchase"] for s in scat],
            [s["product_name"] for s in scat],
            size=[min(24, 6 + (s["revenue"] or 0) / 260) for s in scat],
            color=[s["revenue"] or 0 for s in scat]), 340),
    }
    return page(request, "products.html", "/products", figs=figs, cats=cats,
                top_viewed=top_viewed, leaky=leaky, lowrated=lowrated,
                categories=cat_names)


@router.get("/products/table")
def products_table(request: Request, category: str = "", limit: int = 15):
    sql = ("SELECT product_id, product_name, category, brand, price, rating_avg, "
           "n_view, n_purchases, revenue, view_to_purchase FROM product_stats")
    params = []
    if category:
        sql += " WHERE category = ?"
        params.append(category)
    sql += " ORDER BY revenue DESC, n_view DESC LIMIT ?"
    params.append(min(limit, 100))
    rows = db.query(sql, params)
    top = max([r["n_view"] for r in rows] or [1])
    return templates.TemplateResponse(request, "partials/product_rows.html",
                                      {"rows": rows, "top_view": top})


@router.get("/recommend")
def recommend_page(request: Request):
    sample = db.query(
        "SELECT i.user_id, COUNT(*) n FROM interactions i GROUP BY 1 "
        "HAVING n >= 12 ORDER BY n DESC LIMIT 8")
    cold = db.one(
        "SELECT user_id FROM users WHERE user_id NOT IN "
        "(SELECT DISTINCT user_id FROM interactions) LIMIT 1")
    return page(request, "recommend.html", "/recommend",
                sample=sample, cold_user=cold["user_id"],
                coverage=len(ml.REC["users"]))


@router.post("/recommend/run")
def recommend_run(request: Request, user_id: str = Form(...)):
    user_id = user_id.strip()
    exists = db.one("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
    recs = ml.recommend(user_id, n=5) if exists else []
    seg = ml.segment_of(user_id) if exists else None
    return templates.TemplateResponse(request, "partials/recs.html", {
        "recs": recs, "user_id": user_id, "exists": bool(exists),
        "history": ml.user_history(user_id) if exists else [], "segment": seg,
    })


@router.get("/conversion")
def conversion_page(request: Request):
    opts = {
        "device_type": [r["device_type"] for r in
                        db.query("SELECT DISTINCT device_type FROM sessions ORDER BY 1")],
        "referrer_source": [r["referrer_source"] for r in
                            db.query("SELECT DISTINCT referrer_source FROM sessions ORDER BY 1")],
        "loyalty_tier": [r["loyalty_tier"] for r in
                         db.query("SELECT DISTINCT loyalty_tier FROM users ORDER BY 1")],
        "income_level": [r["income_level"] for r in
                         db.query("SELECT DISTINCT income_level FROM users ORDER BY 1")],
    }
    return page(request, "conversion.html", "/conversion", opts=opts,
                base_rate=ml.CONV_METRICS["base_rate"], m=ml.CONV_METRICS)


@router.post("/conversion/predict")
def conversion_predict(
    request: Request,
    n_view: int = Form(3), n_click: int = Form(1), n_add_to_cart: int = Form(0),
    n_remove_from_cart: int = Form(0), n_add_to_wishlist: int = Form(0),
    n_products: int = Form(3), mean_dwell_ms: int = Form(12000),
    duration_s: int = Form(300), hour: int = Form(14), day_of_week: int = Form(2),
    prior_sessions: int = Form(0), tenure_days: int = Form(180), age: int = Form(35),
    device_type: str = Form("desktop"), referrer_source: str = Form("organic_search"),
    loyalty_tier: str = Form("bronze"), income_level: str = Form("medium"),
):
    prob, factors = ml.predict_conversion(locals())
    return templates.TemplateResponse(request, "partials/conversion_result.html", {
        "prob": prob, "factors": factors, "base_rate": ml.CONV_METRICS["base_rate"],
        "n_add_to_cart": n_add_to_cart,
    })


@router.get("/segments")
def segments_page(request: Request):
    segs = ml.SEG_METRICS["segments"]
    ordered = sorted(segs.items(), key=lambda kv: -kv[1]["size"])

    dist = charts.render(charts.donut(
        [s["name"] for _, s in ordered], [s["size"] for _, s in ordered]), 320)

    # Radar comparison, min-max normalized so differently-scaled features share an axis.
    axes = {"n_views": "viewing", "n_clicks": "clicking", "n_cart_adds": "adding to cart",
            "n_purchases": "buying", "avg_order_value": "spending per order",
            "n_reviews": "writing reviews", "category_diversity": "exploring categories"}
    keys = list(axes)
    mx = {k: max(s["profile"][k] for _, s in ordered) or 1 for k in keys}
    radar = charts.render(charts.radar(
        list(axes.values()),
        {s["name"]: [round(s["profile"][k] / mx[k], 3) for k in keys] for _, s in ordered}), 380)

    return page(request, "segments.html", "/segments", segments=ordered,
                dist=dist, radar=radar, metrics=ml.SEG_METRICS, keys=keys)


@router.get("/model")
def model_page(request: Request):
    m, sm = ml.CONV_METRICS, ml.SEG_METRICS
    figs = {
        "cm": charts.render(charts.heatmap_cm(m["confusion_matrix"]), 300),
        "roc": charts.render(charts.roc(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"],
                                        m["roc_auc"]), 300),
        "importance": charts.render(charts.bar(
            [ml.humanize(f[0]) for f in m["feature_importance"][:12]][::-1],
            [round(f[1], 4) for f in m["feature_importance"][:12]][::-1],
            horizontal=True), 360),
        "elbow": charts.render(charts.elbow(sm["sweep"], sm["k"], sm["rejected_k"]["k"]), 300),
    }
    total_users = db.one("SELECT total_users FROM overview")["total_users"]
    # NB: avoid the key name "items" -- Jinja resolves rec.items to dict.items().
    rec_cov = {
        "n_users": len(ml.REC["users"]),
        "user_share": len(ml.REC["users"]) / total_users,
        "n_cold": total_users - len(ml.REC["users"]),
        "cold_share": (total_users - len(ml.REC["users"])) / total_users,
        "n_items": len(ml.REC["items"]),
        "catalog_share": ml.REC["coverage"]["catalog_share"],
    }
    return page(request, "model.html", "/model", m=m, sm=sm, figs=figs, rec=rec_cov)
