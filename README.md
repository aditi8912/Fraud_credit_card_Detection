# Credit Card Fraud Detection

## Problem Statement
Fraud detection is a core challenge in fintech, where fraud is a rare event (0.17% in this dataset) 
hidden among millions of legitimate transactions. This project builds and compares two classification 
models to detect fraudulent transactions, then evaluates them through a business-cost lens rather than 
just standard ML metrics.

## Dataset
- Source: ULB Credit Card Fraud Detection dataset (Kaggle)
- 284,807 transactions, 492 fraud cases (0.17%)
- Features: Time, Amount, and 28 PCA-anonymized features (V1–V28) for privacy
- `creditcard.csv` is included in this repo via Git LFS
- Note: Prior to this, a synthetic UPI transactions dataset was explored but rejected after 
  correlation analysis showed near-zero relationship (±0.02) between all features and the fraud 
  label — a useful reminder that not every labeled dataset carries real predictive signal.

## Approach
1. Data cleaning: removed 1,081 duplicate rows, verified zero nulls
2. Scaled `Time` and `Amount` (StandardScaler) — other features were already PCA-transformed
3. Stratified 80/20 train-test split to preserve class imbalance in both sets
4. Trained two models with `class_weight='balanced'` to handle extreme imbalance:
   - Logistic Regression (interpretable baseline)
   - Random Forest (100 trees)
5. Evaluated using precision, recall, F1, and ROC-AUC — not accuracy, since 99.8% accuracy is 
   trivially achievable by predicting "not fraud" every time

## Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Recall (fraud caught) | 92% (90/98) | 76% (74/98) |
| Precision (fraud) | 6% | 96% |
| False alarms | 1,389 | 3 |
| Missed fraud | 8 | 24 |
| ROC-AUC | 0.972 | 0.958 |

## Business-Cost Analysis
Using illustrative cost assumptions (missed fraud ≈ ₹5,000, false alarm ≈ ₹50 in review/friction cost):
- Logistic Regression estimated cost: ₹109,450
- Random Forest estimated cost: ₹120,150

Despite Random Forest's much higher precision, Logistic Regression is the cheaper model under this 
cost structure — because missing fraud is far more expensive than a false alarm, and LR misses fewer 
fraud cases. This illustrates why model selection in fraud detection should be cost-driven, not 
accuracy-driven.

## Threshold Tuning
Default threshold (0.5) gives high recall but very low precision. Tuning to 0.95:
- Precision improved from 6% → 39%
- Recall dropped only slightly: 91.8% → 88.8%
- Roughly a 6x reduction in false alarms for near-identical fraud-catch rate

## Key Feature Drivers (Random Forest)
Top predictors: V14, V10, V4, V17, V12 — consistent with the features showing strongest linear 
correlation to fraud in EDA. Features are PCA-anonymized, so real-world interpretation isn't possible, 
but the consistency across two independent methods (correlation + tree importance) supports confidence 
in these signals.

## Dashboard
![Dashboard](<img width="1919" height="1023" alt="Dashboard" src="https://github.com/user-attachments/assets/e1f3a045-641d-4873-b5e5-6bef4f107272" />


Interactive Tableau workbook included: `Charts.twbx` (open in Tableau Desktop or Tableau Public Desktop)

## Limitations
- PCA-anonymized features prevent real-world interpretation of *why* a transaction looks fraudulent
- Cost assumptions in the business analysis are illustrative, not sourced from real fintech data
- Dataset is from 2013 and may not reflect current fraud patterns

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, Tableau
