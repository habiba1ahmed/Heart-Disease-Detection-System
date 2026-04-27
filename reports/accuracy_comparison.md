# Accuracy Comparison Report

## Validation Summary

- Decision Tree test size: `205`
- Expert System test size: `205`

## Metrics Comparison

| Metric | Decision Tree | Expert System |
|---|---:|---:|
| Accuracy | 0.9805 | 0.5415 |
| Precision | 1.0000 | 0.5285 |
| Recall | 0.9619 | 0.9714 |
| F1-score | 0.9806 | 0.6846 |

## Decision Tree Configuration

- Best CV F1: `0.9869`
- Best parameters: `{'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}`
- Overfitting gap (train F1 - test F1): `0.0194`

## Explainability Notes

- Expert system decisions are human-readable through explicit rules and matched-rule traces.
- Decision tree decisions are data-driven and usually yield higher predictive performance.
- The project benefits from both: transparent rule reasoning plus stronger ML accuracy.

## Generated From

- `C:\ANU\Intellegent Programming\Heart_Diease_Detection\reports\ml_metrics.json`
- `C:\ANU\Intellegent Programming\Heart_Diease_Detection\reports\expert_metrics.json`
