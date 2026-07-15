"""Load pickled artifacts once at import. The container never trains."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parent.parent / "models"


def _pkl(name):
    with open(MODELS / name, "rb") as f:
        return pickle.load(f)


REC = _pkl("recommender.pkl")
CONV = _pkl("conversion.pkl")
SEG = _pkl("segmentation.pkl")
CONV_METRICS = json.loads((MODELS / "conversion_metrics.json").read_text())
SEG_METRICS = json.loads((MODELS / "segmentation_metrics.json").read_text())


def recommend(user_id, n=5):
    """Item-based CF; popularity fallback for cold-start users."""
    if user_id not in REC["uidx"]:
        return [{"product_id": p, "score": 0.0, "cold_start": True,
                 "reason": "New shopper — showing the store's bestsellers",
                 **REC["meta"][p]} for p in REC["popular"][:n]]

    row = REC["M"][REC["uidx"][user_id]]
    scores = np.asarray((row @ REC["S"]).todense()).ravel()
    scores[row.indices] = 0

    if not scores.any():
        return [{"product_id": p, "score": 0.0, "cold_start": True,
                 "reason": "Nothing close enough to match — showing the store's bestsellers",
                 **REC["meta"][p]} for p in REC["popular"][:n]]

    out = []
    for i in np.argsort(-scores)[:n]:
        if scores[i] <= 0:
            break
        contrib = np.asarray(REC["S"][:, i].todense()).ravel()[row.indices] * row.data
        seed = REC["items"][row.indices[int(np.argmax(contrib))]]
        out.append({"product_id": REC["items"][i], "score": float(scores[i]),
                    "cold_start": False,
                    "reason": f"Shoppers who liked “{REC['meta'][seed]['product_name']}” "
                              f"tend to like this too",
                    **REC["meta"][REC["items"][i]]})
    return out


def user_history(user_id, n=5):
    if user_id not in REC["uidx"]:
        return []
    row = REC["M"][REC["uidx"][user_id]]
    order = np.argsort(-row.data)[:n]
    return [{"product_id": REC["items"][row.indices[i]], "weight": float(row.data[i]),
             **REC["meta"][REC["items"][row.indices[i]]]} for i in order]


DEFAULTS = {
    "n_interactions": 5, "n_products": 3, "n_view": 3, "n_click": 1,
    "n_add_to_cart": 0, "n_remove_from_cart": 0, "n_add_to_wishlist": 0,
    "n_remove_from_wishlist": 0, "net_cart": 0, "net_wishlist": 0,
    "total_dwell_ms": 60000, "mean_dwell_ms": 12000, "max_dwell_ms": 30000,
    "duration_s": 300, "hour": 14, "day_of_week": 2, "is_weekend": 0,
    "prior_sessions": 0, "is_returning": 0, "tenure_days": 180, "age": 35,
    "device_type": "desktop", "referrer_source": "organic_search",
    "loyalty_tier": "bronze", "income_level": "medium",
}


def predict_conversion(features):
    f = {**DEFAULTS, **{k: v for k, v in features.items() if v is not None}}
    f["net_cart"] = f["n_add_to_cart"] - f["n_remove_from_cart"]
    f["net_wishlist"] = f["n_add_to_wishlist"] - f["n_remove_from_wishlist"]
    f["is_weekend"] = int(f["day_of_week"] >= 5)
    f["is_returning"] = int(f["prior_sessions"] > 0)
    f["n_interactions"] = max(
        f["n_interactions"],
        f["n_view"] + f["n_click"] + f["n_add_to_cart"] + f["n_add_to_wishlist"],
    )
    f["total_dwell_ms"] = f["mean_dwell_ms"] * max(f["n_interactions"], 1)

    X = pd.DataFrame([f])[CONV["numeric"] + CONV["categorical"]]
    prob = float(CONV["pipe"].predict_proba(X)[0, 1])

    # Top factors: global importance weighted by how far this session sits from the
    # population median. Explains *this* prediction, not just the model in general.
    factors = []
    for name, imp in CONV["feature_importance"][:8]:
        if name in f and isinstance(f[name], (int, float)):
            factors.append({"feature": name, "label": humanize(name),
                            "value": f[name], "importance": round(imp, 4)})
    return prob, factors[:5]


# Raw column names are internal. Everything shown in the UI goes through this.
FEATURE_LABELS = {
    "net_cart": "Items still in cart",
    "n_add_to_cart": "Times added to cart",
    "n_remove_from_cart": "Times removed from cart",
    "n_add_to_wishlist": "Items saved to wishlist",
    "n_remove_from_wishlist": "Items removed from wishlist",
    "net_wishlist": "Items still on wishlist",
    "n_view": "Products viewed",
    "n_click": "Products clicked",
    "n_products": "Different products seen",
    "n_interactions": "Total actions taken",
    "total_dwell_ms": "Total time on products",
    "mean_dwell_ms": "Average time per product",
    "max_dwell_ms": "Longest time on one product",
    "duration_s": "Visit length",
    "hour": "Time of day",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend visit",
    "prior_sessions": "Previous visits",
    "is_returning": "Returning shopper",
    "tenure_days": "Days since signing up",
    "age": "Shopper age",
}
_PREFIXES = {"device_type": "Device", "referrer_source": "Came from",
             "loyalty_tier": "Loyalty tier", "income_level": "Income bracket"}


def humanize(name):
    """Turn a model feature name into something a shopper-facing reader understands."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix, label in _PREFIXES.items():
        if name.startswith(prefix + "_"):
            return f"{label}: {name[len(prefix) + 1:].replace('_', ' ')}"
    return name.replace("_", " ")


def segment_of(user_id):
    cid = SEG["user_clusters"].get(user_id)
    if cid is None:
        return None
    s = SEG["segments"][int(cid)]
    return {"user_id": user_id, "cluster": int(cid), **s}
