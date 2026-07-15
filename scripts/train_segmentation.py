"""Behavioral K-Means segmentation over all 10,000 users.

Scope choice: all users are segmented, not just the 6,944 with interactions. The
3,056 zero-history users are a real commercial segment (dormant signups), and
dropping them would hide a third of the base. They cluster out on their own.
"""
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DB = "data/recopulse.db"
OUT = Path("models/segmentation.pkl")
METRICS = Path("models/segmentation_metrics.json")

FEATURES = ["n_views", "n_clicks", "n_cart_adds", "n_wishlist_adds", "n_purchases",
            "total_spend", "avg_order_value", "n_reviews", "avg_rating_given",
            "category_diversity", "n_sessions", "conversion_rate"]
K_RANGE = range(2, 9)


def user_features(con):
    users = pd.read_sql("SELECT user_id, age FROM users", con).set_index("user_id")

    piv = pd.read_sql(
        "SELECT user_id, interaction_type, COUNT(*) n FROM interactions GROUP BY 1,2", con
    ).pivot(index="user_id", columns="interaction_type", values="n").fillna(0)
    piv = piv.rename(columns={"view": "n_views", "click": "n_clicks",
                              "add_to_cart": "n_cart_adds", "add_to_wishlist": "n_wishlist_adds"})

    div = pd.read_sql(
        "SELECT i.user_id, COUNT(DISTINCT p.category) category_diversity "
        "FROM interactions i JOIN products p USING(product_id) GROUP BY 1", con
    ).set_index("user_id")

    pur = pd.read_sql(
        "SELECT user_id, COUNT(*) n_purchases, SUM(total_amount) total_spend "
        "FROM purchases GROUP BY 1", con).set_index("user_id")
    aov = pd.read_sql(
        "SELECT user_id, AVG(order_total) avg_order_value FROM ("
        "  SELECT user_id, order_id, SUM(total_amount) order_total "
        "  FROM purchases GROUP BY 1,2) GROUP BY 1", con).set_index("user_id")

    rev = pd.read_sql(
        "SELECT user_id, COUNT(*) n_reviews, AVG(rating) avg_rating_given "
        "FROM reviews GROUP BY 1", con).set_index("user_id")

    ses = pd.read_sql(
        "SELECT user_id, COUNT(*) n_sessions, AVG(is_converted) conversion_rate "
        "FROM sessions GROUP BY 1", con).set_index("user_id")

    f = users.join([piv, div, pur, aov, rev, ses])
    for c in FEATURES:
        if c not in f:
            f[c] = 0.0
    # Zero-fill is semantically correct here: no purchase row == zero purchases.
    # avg_rating_given has no natural zero, so absent reviewers get the global mean.
    f["avg_rating_given"] = f["avg_rating_given"].fillna(f["avg_rating_given"].mean())
    return f[FEATURES].fillna(0.0)


def name_segment(p, overall):
    """Name clusters from their profile relative to the population."""
    if p["n_sessions"] == 0 and p["n_views"] == 0:
        return "Dormant Signups", "Registered but never browsed — zero sessions, zero interactions"
    if p["n_purchases"] >= overall["n_purchases"] * 2 and p["avg_order_value"] >= overall["avg_order_value"]:
        return "High-Value Buyers", "Convert often at above-average order values — the revenue core"
    if p["n_purchases"] >= overall["n_purchases"]:
        return "Converting Regulars", "Buy steadily at typical basket sizes"
    if p["n_views"] >= overall["n_views"] * 1.4 and p["n_purchases"] < overall["n_purchases"]:
        return "Heavy Browsers", "High engagement, little conversion — the biggest untapped upside"
    if p["n_wishlist_adds"] >= overall["n_wishlist_adds"] * 1.3:
        return "Wishlist Collectors", "Save intent heavily but rarely check out"
    if p["n_views"] < overall["n_views"] * 0.6:
        return "Low-Touch Visitors", "Brief, shallow sessions with minimal engagement"
    return "Casual Shoppers", "Moderate browsing, occasional purchase"


def main():
    con = sqlite3.connect(DB)
    f = user_features(con)
    con.close()
    print(f"users={len(f):,}  features={len(FEATURES)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(f)

    print("\n=== choosing k (elbow + silhouette) ===")
    print("   k    inertia    silhouette")
    sweep = []
    rng = np.random.default_rng(42)
    sub = rng.choice(len(X), 5000, replace=False)  # silhouette is O(n^2); sample it
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X[sub], km.labels_[sub])
        sweep.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
        print(f"  {k:2d}  {km.inertia_:9.0f}    {sil:.4f}")

    # k=2 takes the global silhouette max (0.564) but is a degenerate "engaged vs not"
    # split -- it carries no commercial information and buries the dormant cohort inside
    # a generic bucket. Silhouette always favours the trivial split on data shaped like
    # this, so we exclude k=2 and take the local silhouette maximum among the remaining
    # k, which also lands past the inertia elbow. Both numbers are reported on /model.
    global_best = max(sweep, key=lambda s: s["silhouette"])
    candidates = [s for s in sweep if s["k"] >= 3]
    best = max(candidates, key=lambda s: s["silhouette"])
    best_k = best["k"]
    print(f"\n  global silhouette max: k={global_best['k']} ({global_best['silhouette']:.4f}) "
          f"-- REJECTED as a degenerate active/dormant split")
    print(f"  -> chose k={best_k} (best non-degenerate silhouette = {best['silhouette']:.4f})")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    f["cluster"] = km.labels_
    sil = silhouette_score(X[sub], km.labels_[sub])

    prof = f.groupby("cluster")[FEATURES].mean()
    overall = f[FEATURES].mean()
    sizes = f.cluster.value_counts().sort_index()

    print("\n=== segment profiles ===")
    segments = {}
    used = set()
    for c in prof.index:
        nm, desc = name_segment(prof.loc[c], overall)
        while nm in used:  # keep names unique if two clusters profile alike
            nm += " II"
        used.add(nm)
        segments[int(c)] = {
            "name": nm, "description": desc,
            "size": int(sizes[c]), "share": float(sizes[c] / len(f)),
            "profile": {k: float(v) for k, v in prof.loc[c].items()},
        }
        p = prof.loc[c]
        print(f"\n  [{c}] {nm}  — {sizes[c]:,} users ({sizes[c]/len(f):.1%})")
        print(f"      {desc}")
        print(f"      views={p.n_views:6.1f} clicks={p.n_clicks:5.1f} carts={p.n_cart_adds:5.1f} "
              f"purch={p.n_purchases:4.2f} AOV={p.avg_order_value:7.2f} "
              f"reviews={p.n_reviews:4.2f} cat_div={p.category_diversity:4.2f}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f_:
        pickle.dump({"scaler": scaler, "kmeans": km, "features": FEATURES,
                     "segments": segments,
                     "user_clusters": f["cluster"].to_dict()}, f_)
    METRICS.write_text(json.dumps(
        {"k": int(best_k), "silhouette": float(sil), "sweep": sweep,
         "rejected_k": {"k": int(global_best["k"]), "silhouette": global_best["silhouette"],
                        "reason": "Degenerate engaged/dormant split — higher silhouette but "
                                  "no actionable structure."},
         "segments": segments, "features": FEATURES}, indent=2))
    print(f"\nsaved {OUT} + {METRICS}")


if __name__ == "__main__":
    main()
