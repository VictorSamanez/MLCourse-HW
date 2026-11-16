---
title: Bank Term Deposit Prediction
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
---

# Bank Term Deposit Prediction

FastAPI service that estimates the probability that a bank client subscribes to a term
deposit after a marketing campaign. The model is an XGBoost classifier trained on the UCI
Bank Marketing dataset (`bank-full.csv`, 45,211 rows). `duration` is excluded because it
leaks the target.

## Endpoints

- `GET /health` - status, model checksum and holdout metrics
- `GET /docs` - Swagger UI
- `POST /predict` - client features in, probability and decision out

Example request:

```json
{
  "age": 41,
  "job": "technician",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 1200,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "may",
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

Example response:

```json
{"subscription_probability": 0.100903, "subscription": false, "threshold": 0.23}
```

Holdout ROC-AUC 0.8069, PR-AUC 0.4719, F1 0.500 at threshold 0.23. Predictions assume the
2008-2010 Portuguese retail banking distribution; recent data drifts significantly.
