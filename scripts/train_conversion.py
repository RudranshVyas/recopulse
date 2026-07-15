"""Session-level conversion model.

Honesty notes (see data_profile.md 3.2-3.4):
  * The `purchases` table is a PERFECT leak (a session is converted iff it has a
    purchase row). Nothing derived from it is used here.
  * `add_to_cart` is a one-way separator: 0 carts -> 0 conversions across 10,944
    sessions. It is KEPT (it causally precedes purchase and is observable
    mid-session) but it dominates the model. To show exactly how much, we also
    train an ABLATION with all cart features removed and report both.
"""
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB = "data/recopulse.db"
OUT = Path("models/conversion.pkl")
METRICS = Path("models/conversion_metrics.json")

NUMERIC = ["n_interactions", "n_products", "n_view", "n_click", "n_add_to_cart",
           "n_remove_from_cart", "n_add_to_wishlist", "n_remove_from_wishlist",
           "net_cart", "net_wishlist", "total_dwell_ms", "mean_dwell_ms", "max_dwell_ms",
           "duration_s", "hour", "day_of_week", "is_weekend", "prior_sessions",
           "is_returning", "tenure_days", "age"]
CATEGORICAL = ["device_type", "referrer_source", "loyalty_tier", "income_level"]
CART_FEATURES = ["n_add_to_cart", "n_remove_from_cart", "net_cart"]
TARGET = "is_converted"


def build(numeric, categorical, model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(pipe, Xte, yte, name):
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "name": name,
        "roc_auc": roc_auc_score(yte, proba),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "report": classification_report(yte, pred, target_names=["not converted", "converted"],
                                        output_dict=True, zero_division=0),
        "proba": proba,
        "pred": pred,
    }


def show(m):
    cm = m["confusion_matrix"]
    print(f"\n  {m['name']}")
    print(f"    ROC-AUC   {m['roc_auc']:.4f}")
    print(f"    precision {m['precision']:.4f}   recall {m['recall']:.4f}   F1 {m['f1']:.4f}")
    print( "    confusion matrix        pred_no   pred_yes")
    print(f"                 actual_no  {cm[0][0]:7,}   {cm[0][1]:7,}")
    print(f"                 actual_yes {cm[1][0]:7,}   {cm[1][1]:7,}")


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM session_features", con)
    con.close()

    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].astype(int)
    print(f"sessions={len(df):,}  positives={y.sum():,}  base rate={y.mean():.4f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"train={len(Xtr):,}  test={len(Xte):,} (stratified)")

    print("\n=== FULL FEATURE SET ===")
    results = []
    for name, model in [
        ("LogisticRegression (baseline)",
         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("RandomForest",
         RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                class_weight="balanced", random_state=42)),
    ]:
        pipe = build(NUMERIC, CATEGORICAL, model)
        pipe.fit(Xtr, ytr)
        m = evaluate(pipe, Xte, yte, name)
        m["pipe"] = pipe
        show(m)
        results.append(m)

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n  -> best: {best['name']} (ROC-AUC {best['roc_auc']:.4f})")

    # --- ablation: how much of the model is just "did they add to cart?" ---
    print("\n=== ABLATION: cart features removed ===")
    print("    (isolates whatever signal exists beyond the generator's cart rule)")
    num_abl = [c for c in NUMERIC if c not in CART_FEATURES]
    abl_pipe = build(num_abl, CATEGORICAL,
                     RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                            class_weight="balanced", random_state=42))
    abl_pipe.fit(Xtr[num_abl + CATEGORICAL], ytr)
    abl = evaluate(abl_pipe, Xte[num_abl + CATEGORICAL], yte, "RandomForest (no cart features)")
    show(abl)

    # --- feature importance from the best model ---
    pre = best["pipe"].named_steps["pre"]
    names = NUMERIC + list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    clf = best["pipe"].named_steps["clf"]
    imp = (clf.feature_importances_ if hasattr(clf, "feature_importances_")
           else np.abs(clf.coef_[0]))
    fi = sorted(zip(names, imp.tolist()), key=lambda t: -t[1])

    print("\n=== FEATURE IMPORTANCE (top 15) ===")
    for n, v in fi[:15]:
        print(f"    {n:34s} {v:.4f}  {'#' * int(v * 120)}")
    print("\n  audit cross-check -- features the audit predicted are noise:")
    for n, v in fi:
        if n.startswith(("device_type", "referrer_source")) or "dwell" in n:
            print(f"    {n:34s} {v:.4f}")

    fpr, tpr, _ = roc_curve(yte, best["proba"])
    payload = {
        "best_model": best["name"],
        "base_rate": float(y.mean()),
        "n_train": len(Xtr), "n_test": len(Xte),
        "roc_auc": best["roc_auc"], "precision": best["precision"],
        "recall": best["recall"], "f1": best["f1"],
        "confusion_matrix": best["confusion_matrix"],
        "report": best["report"],
        "roc_curve": {"fpr": fpr[::5].tolist(), "tpr": tpr[::5].tolist()},
        "feature_importance": fi[:15],
        "baseline": {k: results[0][k] for k in ("name", "roc_auc", "precision", "recall", "f1")},
        "ablation": {k: abl[k] for k in ("name", "roc_auc", "precision", "recall", "f1")},
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"pipe": best["pipe"], "numeric": NUMERIC,
                     "categorical": CATEGORICAL, "feature_importance": fi}, f)
    METRICS.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {OUT} + {METRICS}")


if __name__ == "__main__":
    main()
