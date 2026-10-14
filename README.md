# Freight Rate Prediction: Spotter ML Engineer Assessment

This repo predicts `posted_rate` for truckload freight. It uses a LightGBM model on log rate-per-mile, validated with a time-based split.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/train.py      # EDA summary, validation metrics, final model, both prediction files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/make_report.py  # builds report.pdf (optional)
```

## Outputs
- `validation_predictions.csv`: 12,000 rows, `load_id,predicted_rate`
- `data/december_chart_inputs.csv`: the 31 December rows with `predicted_rate` filled in
- `scorer_results/candidate_december.png`: chart produced by the provided `score.py`
- `outputs/metrics.json`: EDA stats, holdout metrics and feature importance
- `report.pdf`: write-up

## Layout
- `src/features.py`: cleaning (weight sign fixes and imputation, outlier removal) and feature engineering
- `src/train.py`: time split, baseline vs LightGBM, final training, predictions
- `src/make_report.py`: PDF report

## Results (time holdout: train Jan–Aug, test Sep–Oct 2025)
| Model | MAE ($) | MAPE |
|---|---|---|
| Lane + equipment median $/mile (baseline) | 93.6 | 3.89% |
| LightGBM | 58.7 | 2.30% |

These metrics are on the cleaned holdout. On the raw holdout, including the outlier rows, the baseline scores MAE 145.6 and LightGBM 109.7.
