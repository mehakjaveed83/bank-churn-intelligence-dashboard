# Project Agent Instructions

This project uses an agent that is especially good at getting knowledge work done. The agent should approach the work like a practical analytics partner: clarify the business question, inspect the available data, build useful models, explain the results clearly, and turn the work into something decision-makers can use.

## Project Focus

This project is focused on churn analysis and an end-to-end analytics workflow:

1. EDA
2. Clustering
3. Prediction
4. SHAP Explainability
5. AI Recommendation Engine
6. Streamlit Dashboard

## Working Style

- Prefer clear, reproducible analysis over one-off notebook experiments.
- Start by understanding the dataset, target variable, feature meanings, data quality, and business context.
- Keep the workflow organized so each stage can feed the next stage cleanly.
- Explain findings in plain language, especially when describing customer segments, churn drivers, model behavior, and recommended actions.
- When building code, favor readable Python, small functions, documented assumptions, and outputs that can be reused in the Streamlit dashboard.

## Analytics Workflow

### 1. EDA

- Profile the dataset for shape, missing values, duplicates, feature types, outliers, class balance, and obvious leakage.
- Summarize churn behavior across important customer attributes.
- Produce visualizations that reveal patterns rather than decoration.
- Capture key insights that should influence feature engineering, clustering, and modeling.

### 2. Clustering

- Prepare customer-level features suitable for segmentation.
- Scale or encode features as needed.
- Evaluate multiple cluster counts and document the reasoning for the selected approach.
- Interpret clusters as actionable customer groups, not just numeric labels.

### 3. Prediction

- Build supervised models to predict churn or the project target.
- Use appropriate train/test splitting and validation.
- Compare baseline and stronger models using suitable metrics such as accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices.
- Pay special attention to class imbalance and business cost tradeoffs.

### 4. SHAP Explainability

- Use SHAP to explain global feature importance and individual predictions.
- Translate model drivers into business language.
- Highlight factors that increase or reduce churn risk.
- Avoid presenting model explanations as causal claims unless the analysis supports that conclusion.

### 5. AI Recommendation Engine

- Convert churn risk, cluster membership, and SHAP explanations into practical recommendations.
- Recommend targeted retention actions for customer groups and individual customers.
- Keep recommendations grounded in model evidence and business constraints.
- Make the recommendation logic transparent enough to audit and improve.

### 6. Streamlit Dashboard

- Build a dashboard that helps users explore the data, segments, predictions, explanations, and recommendations.
- Prioritize useful controls, clear metrics, readable charts, and direct decision support.
- Include views for overview analytics, customer segments, churn prediction, SHAP explanations, and recommended actions.
- Keep the interface polished, fast enough for interactive use, and easy to run locally.

## Deliverable Expectations

- The final project should feel like a complete knowledge-work system, not just a model.
- Analysis should lead naturally into segmentation, prediction, explainability, recommendations, and the dashboard.
- Every major result should answer: what did we learn, why does it matter, and what should be done next?
