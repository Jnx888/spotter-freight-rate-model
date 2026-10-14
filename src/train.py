"""End-to-end pipeline: EDA summary, time-based validation, baseline vs LightGBM,
final training, and prediction files.  Run:  python src/train.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CATEGORICALS, FEATURES, add_features, clean_inputs, remove_rate_outliers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_estimators=800,
              random_state=42, verbose=-1)


def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE_%": float(np.mean(np.abs(err) / y_true) * 100)}


def fit_model(train: pd.DataFrame) -> lgb.LGBMRegressor:
    # Target = log(rate per mile): makes long and short hauls comparable and reduces skew.
    y = np.log(train["posted_rate"] / train["distance"])
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(train[FEATURES], y, categorical_feature=CATEGORICALS)
    return model


def predict(model, df):
    return np.exp(model.predict(df[FEATURES])) * df["distance"].to_numpy()


def baseline_fit(train):
    """Baseline: median rate-per-mile per lane+equipment (fallback: per equipment)."""
    rpm = train["posted_rate"] / train["distance"]
    lane = rpm.groupby([train["pickup"], train["delivery"], train["equipment"]], observed=True).median()
    equip = rpm.groupby(train["equipment"], observed=True).median()
    return lane, equip


def baseline_predict(tables, df):
    lane, equip = tables
    keys = pd.MultiIndex.from_frame(df[["pickup", "delivery", "equipment"]].astype(str))
    lane.index = lane.index.set_levels([lvl.astype(str) for lvl in lane.index.levels])
    r = pd.Series(lane.reindex(keys).to_numpy(), index=df.index)
    r = r.fillna(df["equipment"].astype(str).map(equip.rename(index=str)))
    return r.to_numpy() * df["distance"].to_numpy()


def main():
    raw = pd.read_csv(DATA / "train_test.csv")
    weight_median = float(raw["weight"].abs().median())

    # ---- 1. Quick EDA / data-quality summary -----------------------------------------
    rpm = raw["posted_rate"] / raw["distance"]
    eda = {
        "rows": len(raw), "date_range": [raw["date"].min(), raw["date"].max()],
        "missing_weight": int(raw["weight"].isna().sum()),
        "missing_market_index": int(raw["market_index"].isna().sum()),
        "negative_weight": int((raw["weight"] < 0).sum()),
        "weight_at_cap_47500": int((raw["weight"].abs() == 47500).sum()),
        "corr_rate_vs_distance": float(raw["posted_rate"].corr(raw["distance"])),
        "median_rate_per_mile_by_equipment": rpm.groupby(raw["equipment"]).median().round(3).to_dict(),
        "median_rate_by_month": raw.groupby(raw["date"].str[:7])["posted_rate"].median().round(0).to_dict(),
    }

    df = add_features(clean_inputs(raw, weight_median))
    before = len(df)
    df = remove_rate_outliers(df)
    eda["rate_outliers_removed"] = before - len(df)
    print(json.dumps(eda, indent=2))

    # ---- 2. Time-based validation --------------------------------------------------
    # The final predictions are for Nov-Dec (future dates), so we mimic that: train on
    # Jan-Aug, test on Sep-Oct. A random split would leak future market conditions.
    cutoff = pd.Timestamp("2025-09-01")
    tr, te = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    # Evaluate on the CLEANED holdout (main number) and on the raw holdout (incl. outliers).
    te_raw = add_features(clean_inputs(raw, weight_median))
    te_raw = te_raw[te_raw["date"] >= cutoff]

    base = baseline_fit(tr)
    model = fit_model(tr)
    results = {
        "split": f"train {tr['date'].min().date()}..{tr['date'].max().date()} ({len(tr)} rows) | "
                 f"test {te['date'].min().date()}..{te['date'].max().date()} ({len(te)} rows)",
        "baseline_clean_holdout": metrics(te["posted_rate"].to_numpy(), baseline_predict(base, te)),
        "lightgbm_clean_holdout": metrics(te["posted_rate"].to_numpy(), predict(model, te)),
        "baseline_raw_holdout": metrics(te_raw["posted_rate"].to_numpy(), baseline_predict(base, te_raw)),
        "lightgbm_raw_holdout": metrics(te_raw["posted_rate"].to_numpy(), predict(model, te_raw)),
    }
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    results["feature_importance_gain_%"] = (imp / imp.sum() * 100).round(1).sort_values(ascending=False).to_dict()
    print(json.dumps(results, indent=2))
    (OUT / "metrics.json").write_text(json.dumps({"eda": eda, "validation": results}, indent=2))

    # ---- 3. Final model on ALL cleaned data -----------------------------------------
    final = fit_model(df)

    val = add_features(clean_inputs(pd.read_csv(DATA / "validation.csv"), weight_median))
    val["predicted_rate"] = predict(final, val).round(2)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")[["load_id"]]
    out = template.merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # ---- 4. December chart inputs ------------------------------------------------
    # The chart file only has pickup/delivery/distance/equipment/weight/date. We fill the
    # other model inputs from data we already have:
    #   * coordinates: from the city lookup in the training data
    #   * market_index / quote_signal: daily median of that date in validation.csv
    #     (validation.csv covers every December day, so no values are invented)
    dec_path = DATA / "december_chart_inputs.csv"
    dec = pd.read_csv(dec_path)
    cols = list(dec.columns)
    coords = raw.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    x = dec.copy()
    x["pickup_lat"], x["pickup_lon"] = coords.loc[x["pickup"]].to_numpy().T
    x["delivery_lat"], x["delivery_lon"] = coords.loc[x["delivery"]].to_numpy().T
    daily = pd.read_csv(DATA / "validation.csv").groupby("date")[["market_index", "quote_signal"]].median()
    x = x.join(daily, on="date")
    x = add_features(clean_inputs(x, weight_median))
    for col in CATEGORICALS:  # use the same category codes as training
        x[col] = pd.Categorical(x[col].astype(str), categories=df[col].cat.categories)
    dec["predicted_rate"] = predict(final, x).round(2)
    dec[cols].to_csv(dec_path, index=False)
    print("Wrote validation_predictions.csv and filled data/december_chart_inputs.csv")
    print(dec[["date", "predicted_rate"]].describe())


if __name__ == "__main__":
    main()
