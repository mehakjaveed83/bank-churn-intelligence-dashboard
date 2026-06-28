from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from banking_churn_agent import BankingChurnAgent


DATA_PATH = Path("bank-additional/bank-additional-full.csv")

st.set_page_config(
    page_title="Banking Churn Intelligence Agent",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return BankingChurnAgent(DATA_PATH).load_data()


@st.cache_resource(show_spinner=True)
def train_agent(include_duration: bool, n_clusters: int):
    agent = BankingChurnAgent(DATA_PATH)
    df = agent.load_data()
    profile = agent.profile(df)
    clusters = agent.cluster_customers(
        df, include_duration=include_duration, n_clusters=n_clusters
    )
    artifacts = agent.train_prediction_model(df, include_duration=include_duration)
    shap_result = agent.shap_summary(artifacts)
    recommendations = agent.recommendation_table(artifacts)
    return agent, df, profile, clusters, artifacts, shap_result, recommendations


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def plot_target_mix(df: pd.DataFrame) -> go.Figure:
    target = df["y"].value_counts(normalize=True).mul(100).reset_index()
    target.columns = ["outcome", "share"]
    fig = px.pie(
        target,
        names="outcome",
        values="share",
        hole=0.62,
        color="outcome",
        color_discrete_map={"no": "#6a7282", "yes": "#15aabf"},
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
    return fig


def plot_correlation(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(height=530, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def plot_unknowns(profile: dict) -> go.Figure:
    unknowns = (
        pd.Series(profile["unknown_counts"], name="unknown_rows")
        .sort_values(ascending=True)
        .reset_index()
    )
    unknowns.columns = ["feature", "unknown_rows"]
    fig = px.bar(
        unknowns,
        x="unknown_rows",
        y="feature",
        orientation="h",
        color="unknown_rows",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def plot_numeric_distribution(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure()
    for outcome, color in [("no", "#6a7282"), ("yes", "#15aabf")]:
        values = df.loc[df["y"] == outcome, feature]
        fig.add_trace(
            go.Violin(
                y=values,
                name=outcome,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color=color,
                opacity=0.72,
            )
        )
    fig.update_layout(
        title=f"{feature} distribution by outcome",
        height=430,
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis_title=feature,
    )
    return fig


def plot_box_by_category(df: pd.DataFrame, category: str, numeric: str) -> go.Figure:
    ordered = (
        df.groupby(category)[numeric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig = px.box(
        df,
        x=category,
        y=numeric,
        color="y",
        category_orders={category: ordered},
        points=False,
        color_discrete_map={"no": "#8792a2", "yes": "#14b8a6"},
    )
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=35, b=80),
        xaxis_tickangle=-35,
    )
    return fig


def plot_cluster_map(clustered: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        clustered,
        x="pca_1",
        y="pca_2",
        color="cluster",
        symbol="target",
        hover_data=["age", "job", "marital", "education", "campaign", "previous"],
        opacity=0.78,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def plot_confusion_matrix(matrix: list[list[int]]) -> go.Figure:
    fig = px.imshow(
        matrix,
        labels=dict(x="Predicted", y="Actual", color="Customers"),
        x=["no", "yes"],
        y=["no", "yes"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def plot_score_distribution(artifacts) -> go.Figure:
    score_df = pd.DataFrame(
        {
            "subscription_probability": artifacts.y_proba,
            "actual": artifacts.y_test.map({0: "no", 1: "yes"}).values,
        }
    )
    fig = px.histogram(
        score_df,
        x="subscription_probability",
        color="actual",
        marginal="box",
        nbins=45,
        opacity=0.72,
        color_discrete_map={"no": "#8792a2", "yes": "#14b8a6"},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def plot_shap_importance(shap_result: dict) -> go.Figure:
    imp = shap_result["importance"].sort_values("mean_abs_shap", ascending=True)
    fig = px.bar(
        imp,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def plot_shap_dependence(shap_result: dict, feature: str) -> go.Figure:
    values = shap_result["values"][feature]
    sample = shap_result["sample"][feature]
    fig = px.scatter(
        x=sample,
        y=values,
        color=values,
        color_continuous_scale="RdBu_r",
        labels={"x": feature, "y": "SHAP impact"},
        opacity=0.75,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#52606d")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
    return fig


st.title("Banking Churn Intelligence Agent")
st.caption(
    "EDA -> Clustering -> Prediction -> SHAP Explainability -> AI Recommendations"
)

with st.sidebar:
    st.header("Agent Controls")
    include_duration = st.toggle(
        "Include call duration",
        value=False,
        help="Duration is powerful but leaks post-call information. Keep it off for realistic prediction.",
    )
    n_clusters = st.slider("Customer clusters", 3, 8, 5)
    st.divider()
    st.write("Data source")
    st.code(str(DATA_PATH), language="text")

agent, df, profile, clusters, artifacts, shap_result, recommendations = train_agent(
    include_duration, n_clusters
)

overview, eda, cluster_tab, prediction, explain, recommend = st.tabs(
    [
        "Overview",
        "Advanced EDA",
        "Clustering",
        "Prediction",
        "SHAP Explainability",
        "Recommendations",
    ]
)

with overview:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers", f"{profile['rows']:,}")
    with c2:
        metric_card("Features", f"{profile['columns'] - 1}")
    with c3:
        metric_card(
            "Subscription rate",
            f"{profile['positive_rate']:.1%}",
            "Share of customers with y = yes in the source campaign data.",
        )
    with c4:
        metric_card("Duplicates", f"{profile['duplicates']:,}")

    left, right = st.columns([0.88, 1.12])
    with left:
        st.plotly_chart(plot_target_mix(df), use_container_width=True)
    with right:
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)

    st.info(
        "The source target is term-deposit subscription (`y`). The model predicts `yes` as subscription opportunity, while `no` outcomes provide the churn/non-conversion risk framing for retention analysis."
    )

with eda:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [
        col for col in df.select_dtypes(exclude=np.number).columns.tolist() if col != "y"
    ]

    top, bottom = st.columns([1.1, 0.9])
    with top:
        st.plotly_chart(plot_correlation(df), use_container_width=True)
    with bottom:
        st.plotly_chart(plot_unknowns(profile), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        selected_num = st.selectbox("Violin/box feature", numeric_cols, index=0)
        st.plotly_chart(
            plot_numeric_distribution(df, selected_num), use_container_width=True
        )
    with d2:
        selected_cat = st.selectbox("Category split", cat_cols, index=0)
        box_num = st.selectbox("Box metric", numeric_cols, index=1)
        st.plotly_chart(
            plot_box_by_category(df, selected_cat, box_num), use_container_width=True
        )

with cluster_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Clusters", str(n_clusters))
    with c2:
        metric_card("Silhouette", f"{clusters['silhouette']:.3f}")
    with c3:
        var = sum(clusters["explained_variance"])
        metric_card("PCA variance shown", f"{var:.1%}")

    st.plotly_chart(plot_cluster_map(clusters["clustered"]), use_container_width=True)
    st.dataframe(
        clusters["summary"].style.format(
            {
                "subscription_rate": "{:.1%}",
                "median_age": "{:.0f}",
                "median_campaign": "{:.0f}",
                "previous_contacts": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with prediction:
    m = artifacts.metrics
    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("ROC AUC", f"{m['roc_auc']:.3f}")
    with p2:
        metric_card("Average precision", f"{m['average_precision']:.3f}")
    with p3:
        metric_card("Decision threshold", f"{m['threshold']:.2f}")

    left, right = st.columns([0.85, 1.15])
    with left:
        st.plotly_chart(
            plot_confusion_matrix(m["confusion_matrix"]), use_container_width=True
        )
    with right:
        st.plotly_chart(plot_score_distribution(artifacts), use_container_width=True)

    report = pd.DataFrame(m["classification_report"]).T
    st.dataframe(report, use_container_width=True)

with explain:
    left, right = st.columns([0.95, 1.05])
    with left:
        st.plotly_chart(plot_shap_importance(shap_result), use_container_width=True)
    with right:
        shap_features = shap_result["importance"]["feature"].head(12).tolist()
        feature = st.selectbox("SHAP dependence feature", shap_features)
        st.plotly_chart(
            plot_shap_dependence(shap_result, feature), use_container_width=True
        )

with recommend:
    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card(
            "Top-list mean subscription probability",
            f"{recommendations['subscription_probability'].mean():.1%}",
        )
    with r2:
        metric_card(
            "Critical priority",
            f"{(recommendations['priority'] == 'Critical').sum():,}",
        )
    with r3:
        metric_card("Recommendation rows", f"{len(recommendations):,}")

    priority_filter = st.multiselect(
        "Priority filter",
        ["Critical", "High", "Watch", "Low"],
        default=["Critical", "High"],
    )
    shown = recommendations[
        recommendations["priority"].astype(str).isin(priority_filter)
    ].copy()
    shown["subscription_probability"] = shown["subscription_probability"].map(
        lambda value: f"{value:.1%}"
    )
    st.dataframe(shown, use_container_width=True, hide_index=True)
