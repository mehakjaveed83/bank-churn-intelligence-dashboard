# Bank Churn Intelligence Dashboard

A functional local Streamlit prototype for bank customer churn-style analysis, predictive risk scoring, SHAP explainability, and practical retention recommendations.

## What This Project Uses

The local data in `bank-additional/` is the UCI Bank Marketing dataset with social and economic context. The source target column is `y`, where:

- `yes` means the customer subscribed to the term deposit.
- `no` means the customer did not subscribe.

For this portfolio churn project, the app treats `y = no` as the positive risk class: a churn / non-conversion / retention-risk proxy. This assumption is visible in the dashboard and model artifact.

## Dataset Structure

The full dataset is loaded from:

```text
bank-additional/bank-additional-full.csv
```

Key fields:

- Target: `y`
- Numeric features: `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- Categorical features: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`
- Excluded leakage feature: `duration`

`duration` is excluded because it is only known after the call and would leak outcome information into a pre-call churn model.

## Modeling Approach

The modeling pipeline lives in `src/churn_project/pipeline.py` and includes:

- Median imputation and scaling for numeric features
- Most-frequent imputation and one-hot encoding for categorical features
- A class-balanced random forest classifier
- Stratified train/test split
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix
- Saved model artifact at `artifacts/churn_model.joblib`

## SHAP Explainability

The app uses SHAP TreeExplainer on the trained random forest after preprocessing. It provides:

- Global SHAP feature importance
- Customer-level SHAP contribution chart
- Plain-language recommended actions using predicted risk, customer attributes, and top SHAP drivers

SHAP values are model explanations, not causal proof. They show which features moved this model's prediction up or down.

## Dashboard Views

- Executive Overview: customer counts, risk rate, conversion rate, predicted risk distribution, and segment risk patterns
- Model & SHAP: performance metrics, confusion matrix, model feature importance, and global SHAP importance
- Customer Explainer: individual predicted risk, top SHAP drivers, and next-best retention actions
- Recommendations: portfolio risk bands and retention playbook
- Data Audit: dataset shape, sample rows, unknown categorical values, and model assumptions

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train or refresh the model artifact:

```bash
python train_model.py
```

Run the dashboard:

```bash
streamlit run app.py
```

Then open the localhost URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Smoke Test

Run:

```bash
python tests/smoke_test.py
```

The smoke test checks target handling, leakage exclusion, model training on a sample, SHAP generation, and valid risk probabilities.

## Limitations

- The source dataset is a marketing conversion dataset, not a literal account-closure churn dataset.
- The model predicts non-conversion risk for a term-deposit campaign and frames it as churn/retention risk for the prototype.
- Recommendations are transparent rule-based actions derived from risk score, customer attributes, and SHAP drivers.
- No causal claims are made.
- The app currently uses a static local CSV, not a live customer database.

## Next Steps

- Add clustering for customer segmentation.
- Compare random forest with gradient boosting and calibrated logistic regression.
- Add threshold tuning based on retention budget and contact-capacity constraints.
- Track recommendation outcomes and convert the playbook into a measurable experimentation workflow.
- Add exportable customer action lists for relationship managers.

## Citation

This dataset is described in:

S. Moro, P. Cortez and P. Rita. "A Data-Driven Approach to Predict the Success of Bank Telemarketing." Decision Support Systems, 2014.
