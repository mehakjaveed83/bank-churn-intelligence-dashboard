from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.churn_project.pipeline import (
    ARTIFACT_PATH,
    dataset_profile,
    get_feature_importance,
    individual_explanation,
    load_bank_data,
    load_or_train,
    recommend_actions,
    risk_band,
)


st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #17202a;
        --muted: #5b6470;
        --line: #d9e1ea;
        --panel: #ffffff;
        --risk: #c44536;
        --steady: #177e89;
        --gold: #d49b24;
    }
    .stApp {
        background: linear-gradient(180deg, #f6f8fb 0%, #edf2f7 100%);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #10212f;
    }
    [data-testid="stSidebar"] * {
        color: #eef5f8;
    }
    .main-title {
        font-size: 2.4rem;
        line-height: 1.05;
        font-weight: 760;
        color: #12202f;
        margin-bottom: .2rem;
    }
    .subtle {
        color: #536170;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(25, 41, 61, 0.06);
        min-height: 106px;
    }
    .metric-label {
        color: #617080;
        font-size: .78rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: .04rem;
    }
    .metric-value {
        color: #132536;
        font-size: 1.72rem;
        font-weight: 760;
        margin-top: .2rem;
    }
    .section-title {
        color: #132536;
        font-size: 1.18rem;
        font-weight: 760;
        margin: 1.2rem 0 .45rem 0;
    }
    .action-box {
        background: #ffffff;
        border-left: 5px solid #177e89;
        border-radius: 8px;
        padding: .8rem 1rem;
        margin-bottom: .55rem;
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .7rem .9rem;
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_data() -> pd.DataFrame:
    return load_bank_data(prefer_full=True)


@st.cache_resource(show_spinner="Training or loading the churn model...")
def cached_bundle():
    return load_or_train()


def metric_card(label: str, value: str, helper: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="subtle">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def horizontal_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str) -> go.Figure:
    fig = px.bar(df.sort_values(x), x=x, y=y, orientation="h", title=title)
    fig.update_traces(marker_color=color, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>")
    fig.update_layout(
        height=470,
        margin=dict(l=10, r=15, t=55, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="",
        title_font=dict(size=18),
    )
    return fig


df = cached_data()
bundle = cached_bundle()
profile = dataset_profile(df)

st.sidebar.title("Bank Churn Intelligence")
view = st.sidebar.radio(
    "Dashboard view",
    ["Executive Overview", "Model & SHAP", "Customer Explainer", "Recommendations", "Data Audit"],
)
st.sidebar.divider()
st.sidebar.caption(f"Model artifact: {ARTIFACT_PATH.name}")
st.sidebar.caption("Risk class: y = no")
st.sidebar.caption("Leakage guard: duration excluded")

target_rate = (df["y"].str.lower() == "no").mean()
converted_rate = (df["y"].str.lower() == "yes").mean()
avg_risk = bundle.test_sample["churn_risk"].mean()

st.markdown('<div class="main-title">Bank Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">A local analytics prototype for spotting churn/non-conversion risk, explaining model behavior, and turning evidence into retention actions.</div>',
    unsafe_allow_html=True,
)

if view == "Executive Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers", f"{len(df):,}", "Full bank-additional dataset")
    with c2:
        metric_card("Risk Rate", f"{target_rate:.1%}", "Customers with y = no")
    with c3:
        metric_card("Conversion Rate", f"{converted_rate:.1%}", "Customers with y = yes")
    with c4:
        metric_card("Avg Predicted Risk", f"{avg_risk:.1%}", "Holdout sample")

    left, right = st.columns([1, 1])
    with left:
        churn_counts = df["y"].value_counts().rename_axis("campaign_outcome").reset_index(name="customers")
        fig = px.pie(
            churn_counts,
            values="customers",
            names="campaign_outcome",
            hole=0.55,
            color="campaign_outcome",
            color_discrete_map={"no": "#c44536", "yes": "#177e89"},
            title="Customer Churn / Conversion Distribution",
        )
        fig.update_layout(height=410, margin=dict(l=10, r=10, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        risk_hist = px.histogram(
            bundle.test_sample,
            x="churn_risk",
            nbins=30,
            title="Predicted Risk Distribution",
            color_discrete_sequence=["#d49b24"],
        )
        risk_hist.update_layout(
            height=410,
            xaxis_title="Predicted churn/non-conversion risk",
            yaxis_title="Customers",
            margin=dict(l=10, r=10, t=55, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(risk_hist, use_container_width=True)

    st.markdown('<div class="section-title">Risk by Customer Attribute</div>', unsafe_allow_html=True)
    selected_attribute = st.selectbox(
        "Compare risk rates by",
        ["job", "education", "marital", "contact", "month", "poutcome", "housing", "loan"],
    )
    segment = (
        df.assign(churn_risk_actual=df["y"].str.lower().eq("no"))
        .groupby(selected_attribute, as_index=False)
        .agg(risk_rate=("churn_risk_actual", "mean"), customers=("churn_risk_actual", "size"))
        .query("customers >= 50")
        .sort_values("risk_rate", ascending=False)
    )
    fig = px.bar(
        segment,
        x=selected_attribute,
        y="risk_rate",
        color="customers",
        color_continuous_scale=["#177e89", "#d49b24", "#c44536"],
        title=f"Observed Risk Rate by {selected_attribute}",
        hover_data={"customers": True, "risk_rate": ":.1%"},
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Risk rate", xaxis_title="", height=430)
    st.plotly_chart(fig, use_container_width=True)

elif view == "Model & SHAP":
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", f"{bundle.metrics['roc_auc']:.3f}")
    m2.metric("PR-AUC", f"{bundle.metrics['pr_auc']:.3f}")
    m3.metric("Recall", f"{bundle.metrics['recall']:.3f}")
    m4.metric("Precision", f"{bundle.metrics['precision']:.3f}")
    m5.metric("F1", f"{bundle.metrics['f1']:.3f}")

    cm = bundle.metrics["confusion_matrix"]
    cm_fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Customers"),
        x=["Converted", "Risk"],
        y=["Converted", "Risk"],
        text_auto=True,
        color_continuous_scale=["#f3f7f8", "#177e89", "#c44536"],
        title="Confusion Matrix at 50% Threshold",
    )
    cm_fig.update_layout(height=360)
    st.plotly_chart(cm_fig, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        importance = get_feature_importance(bundle)
        st.plotly_chart(
            horizontal_bar(importance, "importance", "feature", "Random Forest Feature Importance", "#177e89"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            horizontal_bar(bundle.shap_summary, "mean_abs_shap", "feature", "SHAP Global Importance", "#c44536"),
            use_container_width=True,
        )

elif view == "Customer Explainer":
    st.markdown('<div class="section-title">Individual Customer Explainability</div>', unsafe_allow_html=True)
    high_first = bundle.test_sample.sort_values("churn_risk", ascending=False).reset_index(drop=True)
    customer_index = st.slider("Select holdout customer rank by predicted risk", 0, len(high_first) - 1, 0)
    row = high_first.iloc[[customer_index]]
    probability, explanation = individual_explanation(bundle, row)
    band = risk_band(probability)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Predicted Risk", f"{probability:.1%}")
    c2.metric("Risk Band", band)
    c3.dataframe(row[bundle.feature_columns].T.rename(columns={row.index[0]: "value"}), use_container_width=True)

    shap_fig = px.bar(
        explanation.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"raises risk": "#c44536", "reduces risk": "#177e89"},
        title="Top Drivers for Selected Customer",
    )
    shap_fig.update_layout(height=430, xaxis_title="SHAP contribution to risk", yaxis_title="")
    st.plotly_chart(shap_fig, use_container_width=True)

    st.markdown('<div class="section-title">Recommended Next Best Actions</div>', unsafe_allow_html=True)
    for action in recommend_actions(row.iloc[0], probability, explanation):
        st.markdown(f'<div class="action-box">{action}</div>', unsafe_allow_html=True)

elif view == "Recommendations":
    st.markdown('<div class="section-title">Portfolio Retention Playbook</div>', unsafe_allow_html=True)
    scored = bundle.test_sample.copy()
    scored["risk_band"] = pd.cut(
        scored["churn_risk"],
        bins=[0, 0.55, 0.75, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    band_summary = (
        scored.groupby("risk_band", observed=False)
        .agg(customers=("churn_risk", "size"), avg_risk=("churn_risk", "mean"))
        .reset_index()
    )
    fig = px.bar(
        band_summary,
        x="risk_band",
        y="customers",
        color="avg_risk",
        color_continuous_scale=["#177e89", "#d49b24", "#c44536"],
        title="Holdout Customers by Risk Band",
        text="customers",
    )
    fig.update_layout(height=390, xaxis_title="", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)

    playbook = [
        ("High risk", "Use scarce retention budget here first: senior-agent outreach, clearer value proposition, and personalized incentive tests."),
        ("Medium risk", "Use lower-cost nurture flows, improve channel timing, and escalate only when risk drivers persist."),
        ("Low risk", "Protect experience quality: avoid over-contacting and use light-touch education or cross-sell messaging."),
        ("Operational guardrail", "Do not use call duration for pre-call targeting; it is only known after the interaction and would leak outcome information."),
    ]
    for title, body in playbook:
        st.markdown(f'<div class="action-box"><strong>{title}:</strong> {body}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top Customers to Review</div>', unsafe_allow_html=True)
    display_cols = ["churn_risk", "age", "job", "marital", "education", "contact", "campaign", "pdays", "poutcome"]
    st.dataframe(scored.sort_values("churn_risk", ascending=False)[display_cols].head(25), use_container_width=True)

else:
    st.markdown('<div class="section-title">Data Audit & Assumptions</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Rows", f"{profile['rows']:,}")
    a2.metric("Columns", f"{profile['columns']:,}")
    a3.metric("Duplicate Rows", f"{profile['duplicates']:,}")
    a4.metric("Missing Cells", f"{profile['missing_cells']:,}")

    st.write("Dataset structure")
    st.dataframe(df.head(12), use_container_width=True)

    st.write("Categorical `unknown` values")
    unknown_df = pd.DataFrame(
        [{"feature": key, "unknown_rows": value} for key, value in profile["unknown_counts"].items()]
    ).sort_values("unknown_rows", ascending=False)
    st.dataframe(unknown_df, use_container_width=True)

    st.write("Model assumptions")
    st.json(bundle.assumptions)
