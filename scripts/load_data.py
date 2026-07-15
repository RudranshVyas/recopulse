"""CSV -> clean -> SQLite, plus precomputed aggregate tables for serving.

Cleaning rules are documented in README.md. Key decision: products.rating_avg is
NULL iff review_count == 0 (verified structural, not missing) -- kept NULL, never
imputed, so avg-rating KPIs are not dragged toward zero by unrated products.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
DB = Path("data/recopulse.db")

DATE_COLS = {
    "users": ["signup_date"],
    "products": ["date_added"],
    "sessions": ["start_time"],
    "interactions": ["timestamp"],
    "purchases": ["order_date"],
    "reviews": ["review_date"],
}


def clean():
    d = {t: pd.read_csv(RAW / f"{t}.csv") for t in DATE_COLS}

    for t, cols in DATE_COLS.items():
        for c in cols:
            # Timestamps carry sub-microsecond noise ("...000000115"); coerce and floor to seconds.
            d[t][c] = pd.to_datetime(d[t][c], format="mixed").dt.floor("s")

    # Normalize categorical text: strip, collapse case-variants.
    for t, cols in [
        ("users", ["gender", "country", "income_level", "preferred_category", "loyalty_tier"]),
        ("products", ["category", "subcategory", "brand"]),
        ("sessions", ["device_type", "referrer_source"]),
        ("interactions", ["interaction_type"]),
    ]:
        for c in cols:
            d[t][c] = d[t][c].str.strip()

    d["sessions"]["is_converted"] = d["sessions"]["is_converted"].astype(int)

    # Drop the bulky free-text we never render, keep a short blurb for product cards.
    d["products"]["blurb"] = (
        d["products"]["product_description"].str.split("\n").str[0].str.slice(0, 180)
    )
    d["products"] = d["products"].drop(columns=["product_description"])
    d["reviews"]["review_text"] = d["reviews"]["review_text"].str.slice(0, 400)
    d["reviews"]["is_verified"] = d["reviews"]["purchase_id"].notna().astype(int)

    return d


def aggregates(d):
    """Precompute everything the dashboard reads, so no request scans raw tables."""
    prod, inter, pur, rev, sess, users = (
        d["products"], d["interactions"], d["purchases"], d["reviews"], d["sessions"], d["users"]
    )
    out = {}

    # --- per-product rollup: views/clicks/carts/purchases/revenue + conversion ---
    piv = (
        inter.pivot_table(index="product_id", columns="interaction_type",
                          values="interaction_id", aggfunc="count")
        .fillna(0).astype(int)
    )
    piv.columns = [f"n_{c}" for c in piv.columns]
    pstats = pur.groupby("product_id").agg(
        n_purchases=("purchase_id", "count"),
        units_sold=("quantity", "sum"),
        revenue=("total_amount", "sum"),
    )
    ps = (
        prod.set_index("product_id")[["product_name", "category", "subcategory", "brand",
                                      "price", "rating_avg", "review_count", "stock_quantity", "blurb"]]
        .join(piv).join(pstats)
    )
    for c in ps.columns:
        if c.startswith("n_") or c == "units_sold":
            ps[c] = ps[c].fillna(0).astype(int)  # counts stay ints, not 746.0
    ps["revenue"] = ps["revenue"].fillna(0.0)
    ps["view_to_purchase"] = ps.n_purchases / ps.n_view.replace(0, np.nan)
    out["product_stats"] = ps.reset_index()

    # --- per-category rollup ---
    cat = ps.groupby("category").agg(
        products=("product_name", "count"),
        views=("n_view", "sum"),
        carts=("n_add_to_cart", "sum"),
        purchases=("n_purchases", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("price", "mean"),
        avg_rating=("rating_avg", "mean"),  # skipna -> unrated products excluded, not zeroed
    )
    cat["view_to_purchase"] = cat.purchases / cat.views.replace(0, np.nan)
    for c in ["products", "views", "carts", "purchases"]:
        cat[c] = cat[c].astype(int)
    out["category_stats"] = cat.reset_index()

    # --- monthly trend ---
    sm = sess.set_index("start_time").resample("MS").agg(
        sessions=("session_id", "count"), conversions=("is_converted", "sum"))
    pm = pur.set_index("order_date").resample("MS").agg(
        orders=("purchase_id", "count"), revenue=("total_amount", "sum"))
    tr = sm.join(pm).fillna(0)
    tr["conversion_rate"] = tr.conversions / tr.sessions
    out["monthly_trend"] = tr.reset_index().rename(columns={"start_time": "month"})

    # --- headline KPIs (single row) ---
    out["overview"] = pd.DataFrame([{
        "total_users": len(users),
        "total_products": len(prod),
        "total_sessions": len(sess),
        "total_interactions": len(inter),
        "total_purchases": len(pur),
        "conversion_rate": sess.is_converted.mean(),
        "total_revenue": pur.total_amount.sum(),
        "avg_order_value": pur.groupby("order_id").total_amount.sum().mean(),
        "avg_rating": rev.rating.mean(),
        "rated_products": int(prod.rating_avg.notna().sum()),
    }])

    # --- session features (built once here, reused by the conversion trainer) ---
    out["session_features"] = build_session_features(sess, inter, users)

    return out


def build_session_features(sess, inter, users):
    """Session-level features. NOTHING from purchases -- that table is a perfect leak
    (set(purchases.session_id) == set(converted sessions)). See data_profile.md 3.2."""
    piv = (
        inter.pivot_table(index="session_id", columns="interaction_type",
                          values="interaction_id", aggfunc="count")
        .fillna(0).astype(int)
    )
    piv.columns = [f"n_{c}" for c in piv.columns]
    agg = inter.groupby("session_id").agg(
        n_interactions=("interaction_id", "count"),
        n_products=("product_id", "nunique"),
        total_dwell_ms=("dwell_time_ms", "sum"),
        mean_dwell_ms=("dwell_time_ms", "mean"),
        max_dwell_ms=("dwell_time_ms", "max"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
    )
    f = sess.set_index("session_id").join(piv).join(agg)
    f["duration_s"] = (f.last_ts - f.first_ts).dt.total_seconds()
    f["net_cart"] = f.n_add_to_cart - f.n_remove_from_cart
    f["net_wishlist"] = f.n_add_to_wishlist - f.n_remove_from_wishlist
    f["hour"] = f.start_time.dt.hour
    f["day_of_week"] = f.start_time.dt.dayofweek
    f["is_weekend"] = (f.day_of_week >= 5).astype(int)

    # returning user = had an earlier session; tenure = days since signup at session start
    f = f.sort_values("start_time")
    f["prior_sessions"] = f.groupby("user_id").cumcount()
    f["is_returning"] = (f.prior_sessions > 0).astype(int)
    f = f.join(users.set_index("user_id")[["signup_date", "loyalty_tier", "income_level", "age"]],
               on="user_id")
    f["tenure_days"] = (f.start_time - f.signup_date).dt.days

    return f.drop(columns=["first_ts", "last_ts", "signup_date"]).reset_index()


def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()

    d = clean()
    aggs = aggregates(d)

    con = sqlite3.connect(DB)
    for name, df in {**d, **aggs}.items():
        df.to_sql(name, con, index=False, if_exists="replace")
        print(f"  wrote {name:20s} {len(df):>7,} rows")

    for stmt in [
        "CREATE INDEX ix_inter_user ON interactions(user_id)",
        "CREATE INDEX ix_inter_prod ON interactions(product_id)",
        "CREATE INDEX ix_inter_sess ON interactions(session_id)",
        "CREATE INDEX ix_sess_user ON sessions(user_id)",
        "CREATE INDEX ix_pur_user ON purchases(user_id)",
        "CREATE INDEX ix_pstats_cat ON product_stats(category)",
        "CREATE INDEX ix_pstats_id ON product_stats(product_id)",
    ]:
        con.execute(stmt)
    con.commit()

    print("\nverifying row counts against source CSVs:")
    ok = True
    for t in DATE_COLS:
        src = sum(1 for _ in open(RAW / f"{t}.csv", encoding="utf-8")) - 1
        got = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        # CSVs contain embedded newlines in text fields; fall back to pandas for a true count.
        if src != got:
            src = len(pd.read_csv(RAW / f"{t}.csv"))
        mark = "OK " if src == got else "FAIL"
        ok &= src == got
        print(f"  {mark} {t:14s} csv={src:>7,}  db={got:>7,}")
    con.close()
    print("\nDB:", DB, f"({DB.stat().st_size/1e6:.1f} MB)")
    if not ok:
        raise SystemExit("row count mismatch")


if __name__ == "__main__":
    main()
