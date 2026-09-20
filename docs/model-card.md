# Model card

## Intended use

This logistic-regression model demonstrates reproducible churn-risk scoring on synthetic telecom data. It is an educational reference and must not be used for real customer decisions without domain validation, fairness testing, privacy review, and human oversight.

## Data and limitations

- Data is deterministic, synthetic, and contains no customer PII.
- Features model tenure, charges, support activity, contract length, autopay, and services.
- Synthetic performance does not predict production performance.
- A ROC-AUC promotion gate prevents an obviously degraded model from being packaged.

## Monitoring

The project reports feature mean shifts normalized by the training standard deviation. A production implementation should add label-delay handling, segment metrics, calibration, fairness analysis, alerting, and retraining approval.

