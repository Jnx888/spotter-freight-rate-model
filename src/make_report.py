"""Build report.pdf from outputs/metrics.json and the scorer chart."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "outputs/metrics.json").read_text())
e, v = m["eda"], m["validation"]
s = getSampleStyleSheet()
P = lambda t, st="BodyText": Paragraph(t, s[st])
f = lambda d: [f"{d['MAE']:.1f}", f"{d['RMSE']:.1f}", f"{d['MAPE_%']:.2f}%"]
story = [
    P("Freight Rate Prediction: Approach & Validation Report", "Title"),
    P("1. Data overview", "Heading2"),
    P(f"{e['rows']:,} labeled loads from {e['date_range'][0]} to {e['date_range'][1]} across 64 cities and 3 equipment types. "
      f"Rate is driven mainly by distance (correlation {e['corr_rate_vs_distance']:.2f}). Median $/mile by equipment: "
      + ", ".join(f"{k} {val}" for k, val in e['median_rate_per_mile_by_equipment'].items())
      + ". Median rates rise from about $1,915 in January to a June peak of about $2,120, then level off."),
    P("2. Data-quality issues and fixes", "Heading2"),
    P(f"&bull; <b>Negative weights</b> ({e['negative_weight']} rows): magnitudes look like normal loads, so I treated them as sign errors and took the absolute value.<br/>"
      f"&bull; <b>Missing weight</b> ({e['missing_weight']}): filled with the training median.<br/>"
      f"&bull; <b>Weight capped at 47,500 lb</b> ({e['weight_at_cap_47500']} rows): kept and clipped consistently.<br/>"
      f"&bull; <b>Missing market_index</b> ({e['missing_market_index']}): left as missing; LightGBM handles NaN natively.<br/>"
      f"&bull; <b>Rate outliers</b>: about 1.3% of rows have $/mile 2&ndash;7x above or below their lane's median, likely entry or unit errors. "
      f"I removed {e['rate_outliers_removed']} training rows outside 0.5x&ndash;2x of their lane + equipment median. Prediction rows are never dropped."),
    P("3. Split and validation approach", "Heading2"),
    P("The final predictions cover Nov&ndash;Dec 2025, which are future dates the model has never seen. A random split would let the model see "
      "market conditions from the same weeks it is tested on and overstate accuracy. I used a <b>time-based holdout</b>: "
      + v["split"] + ". After choosing the model, I retrained it on all cleaned data (Jan&ndash;Oct) to produce the final predictions."),
    P("4. Model", "Heading2"),
    P("Target: log(rate / distance). Rate per mile makes short and long hauls comparable, and the log reduces skew. "
      "Features: distance, straight-line miles from coordinates, lat/lon, pickup/delivery/equipment as categoricals, weight, "
      "market_index, quote_signal, and calendar fields (day of year, day of week, month). Model: LightGBM gradient boosting "
      "(800 trees, learning rate 0.05, 31 leaves), chosen because it handles mixed categorical and numeric data, missing values and "
      "non-linear interactions with little tuning. Baseline: median $/mile for each lane + equipment, times distance."),
    P("5. Results (Sep&ndash;Oct holdout)", "Heading2"),
    Table([["Model / holdout", "MAE $", "RMSE $", "MAPE"],
           ["Baseline (cleaned)", *f(v["baseline_clean_holdout"])],
           ["LightGBM (cleaned)", *f(v["lightgbm_clean_holdout"])],
           ["Baseline (raw, incl. outliers)", *f(v["baseline_raw_holdout"])],
           ["LightGBM (raw, incl. outliers)", *f(v["lightgbm_raw_holdout"])]]),
    Spacer(1, 6),
    P("LightGBM cuts MAE by about 37% against the baseline. The raw-holdout numbers include the suspected erroneous rows, which no model can predict. "
      "Top features by gain: " + ", ".join(f"{k} {val}%" for k, val in list(v['feature_importance_gain_%'].items())[:5]) + "."),
    P("6. December prediction chart (produced by score.py)", "Heading2"),
    P("Inputs are fixed (Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb); only the date changes. The chart file has no market_index, "
      "quote_signal or coordinates. Coordinates come from the city lookup, and market_index and quote_signal are the daily median for that date in "
      "validation.csv, so no values are invented. Predictions stay in a narrow $821&ndash;$845 band with a weekly pattern."),
    Image(str(ROOT / "scorer_results/candidate_december.png"), width=6.8 * inch, height=3.2 * inch),
    P("7. Limitations", "Heading2"),
    P("Ten months of history cannot show a full-year seasonal pattern, so December seasonality is approximated by the most recent behavior. "
      "With more time I would add lane-level rolling rate features and quantile models for price ranges."),
]
SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch).build(story)
print("Wrote report.pdf")
