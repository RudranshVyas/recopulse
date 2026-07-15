"""Item-based collaborative filtering on implicit feedback.

Weights are mapped to the REAL interaction vocabulary found in the data. There is
no `purchase` interaction_type (see data_profile.md 3.2), so purchase signal is
joined in from the purchases table. Negative actions carry negative weight.
"""
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

DB = "data/recopulse.db"
OUT = Path("models/recommender.pkl")

WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_wishlist": 3.0,
    "add_to_cart": 4.0,
    "purchase": 5.0,           # sourced from purchases table, not interactions
    "remove_from_cart": -2.0,
    "remove_from_wishlist": -1.0,
}
TOP_K = 50  # neighbours retained per item


def main():
    con = sqlite3.connect(DB)
    inter = pd.read_sql("SELECT user_id, product_id, interaction_type FROM interactions", con)
    pur = pd.read_sql("SELECT user_id, product_id FROM purchases", con)
    prods = pd.read_sql(
        "SELECT product_id, product_name, category, brand, price, rating_avg, "
        "n_view, n_purchases FROM product_stats", con)
    con.close()

    pur["interaction_type"] = "purchase"
    events = pd.concat([inter, pur], ignore_index=True)
    events["w"] = events.interaction_type.map(WEIGHTS)

    # Sum weights per (user, item); clip negatives to 0 -- a net-negative signal means
    # "not interested", which in implicit feedback is absence, not a negative preference.
    ui = events.groupby(["user_id", "product_id"]).w.sum().clip(lower=0).reset_index()
    ui = ui[ui.w > 0]

    users = ui.user_id.unique()
    items = prods.product_id.values  # full catalog, so cold items keep a column
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {p: k for k, p in enumerate(items)}

    M = csr_matrix(
        (ui.w.values, (ui.user_id.map(uidx).values, ui.product_id.map(iidx).values)),
        shape=(len(users), len(items)),
    )
    density = M.nnz / (M.shape[0] * M.shape[1])
    print(f"user-item matrix: {M.shape[0]:,} users x {M.shape[1]:,} items, "
          f"nnz={M.nnz:,}, density={density:.4%}")

    # Item-item cosine similarity = normalized(M).T @ normalized(M), kept sparse.
    Mn = normalize(M, norm="l2", axis=0)
    S = (Mn.T @ Mn).tolil()
    S.setdiag(0)
    S = S.tocsr()
    S.eliminate_zeros()

    # Keep only top-K neighbours per item -> bounded memory, less tail noise.
    S = prune_topk(S, TOP_K)
    print(f"item-item similarity: {S.shape[0]}x{S.shape[1]}, nnz={S.nnz:,} (top-{TOP_K} per item)")

    # Popularity fallback for cold-start users (31% of users have no interactions).
    pop = prods.sort_values(["n_purchases", "n_view"], ascending=False).product_id.head(50).tolist()

    art = {
        "M": M, "S": S, "users": users, "items": items,
        "uidx": uidx, "iidx": iidx,
        "popular": pop,
        "weights": WEIGHTS,
        "meta": prods.set_index("product_id").to_dict("index"),
        "density": density,
    }
    art["coverage"] = report(art, prods)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(art, f)
    print(f"\nsaved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


def prune_topk(S, k):
    rows, cols, vals = [], [], []
    for r in range(S.shape[0]):
        lo, hi = S.indptr[r], S.indptr[r + 1]
        d, idx = S.data[lo:hi], S.indices[lo:hi]
        if len(d) > k:
            keep = np.argpartition(-d, k)[:k]
            d, idx = d[keep], idx[keep]
        rows += [r] * len(d)
        cols += idx.tolist()
        vals += d.tolist()
    return csr_matrix((vals, (rows, cols)), shape=S.shape)


def recommend(art, user_id, n=5):
    """Score = sum of similarities to items the user engaged with, weighted by engagement.
    Returns (product_id, score, reason). Falls back to popularity for cold users."""
    if user_id not in art["uidx"]:
        return [(p, 0.0, "Cold start: no interaction history, showing catalog bestsellers")
                for p in art["popular"][:n]]

    row = art["M"][art["uidx"][user_id]]
    scores = np.asarray((row @ art["S"]).todense()).ravel()
    scores[row.indices] = 0  # don't recommend what they already engaged with

    if not scores.any():
        return [(p, 0.0, "No similar items found, showing catalog bestsellers")
                for p in art["popular"][:n]]

    top = np.argsort(-scores)[:n]
    seed_names = {}
    for i in top:
        # attribute the recommendation to the user's item that contributed most
        contrib = np.asarray(art["S"][:, i].todense()).ravel()[row.indices] * row.data
        best = art["items"][row.indices[int(np.argmax(contrib))]]
        seed_names[i] = art["meta"][best]["product_name"]

    return [(art["items"][i], float(scores[i]),
             f"Because you engaged with “{seed_names[i]}”")
            for i in top if scores[i] > 0]


def report(art, prods):
    rng = np.random.default_rng(42)
    sample = rng.choice(art["users"], size=500, replace=False)
    reachable = set()
    served = 0
    for u in sample:
        recs = recommend(art, u, n=10)
        if recs:
            served += 1
        reachable.update(r[0] for r in recs)

    print("\n--- coverage ---")
    print(f"users with CF history:   {len(art['users']):,} / 10,000 "
          f"({len(art['users'])/10000:.1%})  -- rest use popularity fallback")
    print(f"catalog reachable:       {len(reachable):,} / {len(art['items']):,} distinct items "
          f"in top-10 recs across a 500-user sample ({len(reachable)/len(art['items']):.1%})")
    print(f"users served a rec:      {served}/500 ({served/500:.1%})")

    print("\n--- sample recommendations ---")
    for u in list(art["users"][:2]) + ["cold-start-user-not-in-matrix"]:
        print(f"\nuser {u[:8]}...")
        for pid, score, reason in recommend(art, u, n=5):
            m = art["meta"][pid]
            print(f"   {m['product_name'][:42]:44s} {m['category'][:16]:18s} "
                  f"score={score:6.3f}  {reason[:52]}")

    return {"catalog_reachable": len(reachable),
            "catalog_share": len(reachable) / len(art["items"]),
            "sample_size": len(sample)}


if __name__ == "__main__":
    main()
