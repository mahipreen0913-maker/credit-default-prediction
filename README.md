# Credit Default Prediction

A machine learning project predicting whether a credit card client will default
on payment, using the UCI "Default of Credit Card Clients" dataset (30,000
accounts, Taiwan, 2005).

**[Open the notebook in Colab]([[your Colab share link]](https://colab.research.google.com/drive/1wTHgMjmDFR-YkPRcwX9Gr9lY3E_QOnd5?usp=sharing)**

## Objective

Build and compare two classification models — logistic regression and random
forest — to predict client default, with particular attention to the dataset's
class imbalance (~22% default rate) and to which features actually drive risk.

## Data

- Source: UCI Machine Learning Repository, *Default of Credit Card Clients*
- 30,000 observations, 24 features, 1 binary target
- Features: credit limit, demographics, 6 months of repayment status, bill
  amounts, and payment amounts
- Cleaning: folded undocumented category codes in `EDUCATION` and `MARRIAGE`
  into an "other" bucket; confirmed no missing values

## Feature Engineering

Added `MAX_DELAY` — the worst repayment status across the 6-month window —
as a single summary signal of a client's payment history. It ranked as the
2nd most important feature in the final model.

## Models & Results

| Model | Accuracy | Recall (default class) | Precision (default class) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 80.6% | 0.23 | 0.68 | 0.724 |
| Random Forest (balanced) | 77.0% | 0.61 | 0.49 | 0.774 |

**Why accuracy alone is misleading here:** with only ~22% of accounts
defaulting, a model that never predicts default would already score ~78%
accuracy while catching zero real defaulters. The logistic regression's
80.6% accuracy masks a weak 23% recall — it misses more than three-quarters
of actual defaults. Using `class_weight='balanced'` in the random forest
traded a small drop in accuracy for a large recall improvement (23% → 61%),
which is the right trade-off for a credit risk use case, where missing a
real default is typically costlier than a false alarm.

## What Drives Default Risk

The two strongest predictors, by a wide margin, are recent repayment
behavior: `PAY_0` (most recent month's repayment status) and the engineered
`MAX_DELAY` feature — together accounting for ~40% of the random forest's
decision-making. This matches standard credit risk practice: recent payment
behavior is a stronger predictor than static demographics.

Logistic regression coefficients confirm the direction of these effects:
`MAX_DELAY` (+0.48) and `PAY_0` (+0.42) are the strongest positive drivers
of default risk. `BILL_AMT1`, the most recent bill amount, has the largest
negative coefficient (−0.25) — likely reflecting that clients with large,
active balances in good standing are lower-risk than the raw dollar amount
might suggest on its own.

## Limitations

- Dataset reflects Taiwanese credit card clients from 2005; feature
  distributions and default drivers may not transfer directly to other
  markets or time periods.
- `SEX`, `EDUCATION`, and `MARRIAGE` carry non-zero coefficients. A
  production credit model would require a fairness/bias review before any
  real-world deployment.
- Models were not tuned via cross-validation or grid search; results
  reflect a single train/test split with fixed hyperparameters.

## Tools

Python, pandas, scikit-learn, matplotlib, Google Colab
