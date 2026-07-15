"""Profile the raw CSVs and audit signal. Output backs data_profile.md."""
import pandas as pd

RAW = "data/raw"
TABLES = ["users", "products", "sessions", "interactions", "purchases", "reviews"]


def main():
    d = {t: pd.read_csv(f"{RAW}/{t}.csv") for t in TABLES}

    print("=" * 72)
    print("SCHEMA")
    for name, df in d.items():
        print(f"\n{name.upper()}  {df.shape[0]:,} rows x {df.shape[1]} cols")
        nulls = df.isnull().sum()
        for col in df.columns:
            print(f"  {col:22s} {str(df[col].dtype):10s} nulls={nulls[col]:,}")
    print(f"\nTOTAL ROWS: {sum(len(x) for x in d.values()):,}")

    print("\n" + "=" * 72)
    print("REFERENTIAL INTEGRITY (orphan counts, expect 0)")
    fks = [
        ("sessions", "user_id", "users", "user_id"),
        ("interactions", "user_id", "users", "user_id"),
        ("interactions", "session_id", "sessions", "session_id"),
        ("interactions", "product_id", "products", "product_id"),
        ("purchases", "user_id", "users", "user_id"),
        ("purchases", "session_id", "sessions", "session_id"),
        ("purchases", "product_id", "products", "product_id"),
        ("purchases", "interaction_id", "interactions", "interaction_id"),
        ("reviews", "user_id", "users", "user_id"),
        ("reviews", "product_id", "products", "product_id"),
    ]
    for ct, ck, pt, pk in fks:
        orphans = len(set(d[ct][ck].dropna()) - set(d[pt][pk]))
        print(f"  {ct}.{ck:16s} -> {pt+'.'+pk:22s} {orphans}")

    print("\n" + "=" * 72)
    print("SIGNAL AUDIT")
    s, i, p = d["sessions"], d["interactions"], d["purchases"]

    print(f"\nconversion base rate: {s.is_converted.mean():.4f} "
          f"({int(s.is_converted.sum()):,}/{len(s):,})")

    print("\ninteraction_type vocabulary:")
    print(i.interaction_type.value_counts().to_string())

    converted = set(s.loc[s.is_converted, "session_id"])
    print("\nLEAKAGE: set(purchases.session_id) == set(converted sessions)? "
          f"{set(p.session_id) == converted}   <-- purchases excluded from conversion features")

    carts = i[i.interaction_type == "add_to_cart"].groupby("session_id").size()
    m = s.set_index("session_id").join(carts.rename("carts"))
    m["carts"] = m.carts.fillna(0)
    print("\nP(converted) by add_to_cart count (one-way separator check):")
    print(m.groupby(m.carts.clip(0, 4)).is_converted.agg(["mean", "count"]).to_string())

    print("\nno-signal features:")
    for col in ["device_type", "referrer_source"]:
        print(f"\n  {col}:")
        print(s.groupby(col).is_converted.agg(["mean", "count"]).to_string())

    dwell = i.groupby("session_id").dwell_time_ms.mean()
    md = s.set_index("session_id").join(dwell.rename("dwell"))
    print("\n  mean dwell_time_ms by converted:")
    print(md.groupby("is_converted").dwell.mean().to_string())

    print("\ninteraction spread (CF viability):")
    for label, g in [("per session", i.groupby("session_id").size()),
                     ("per user", i.groupby("user_id").size()),
                     ("per product", i.groupby("product_id").size())]:
        print(f"  {label:12s} mean={g.mean():7.2f} median={g.median():5.0f} "
              f"max={g.max():5d} std={g.std():7.2f}")

    print("\ncold-start coverage:")
    print(f"  products with interactions: {i.product_id.nunique():,} / {len(d['products']):,}")
    print(f"  users with interactions:    {i.user_id.nunique():,} / {len(d['users']):,} "
          f"({1 - i.user_id.nunique()/len(d['users']):.0%} need popularity fallback)")


if __name__ == "__main__":
    main()
