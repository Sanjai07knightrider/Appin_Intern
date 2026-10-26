"""UI styling, CSS injection, KPI components, and Plotly chart builders for EduPredict AI."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.constants import (
    CATEGORY_AT_RISK,
    CATEGORY_AVERAGE,
    CATEGORY_BG_COLORS,
    CATEGORY_COLORS,
    CATEGORY_GOOD,
    COL_ASSIGNMENT_AVG,
    COL_ATTENDANCE,
    COL_CONFIDENCE,
    COL_INTERNAL_MARKS,
    COL_PERFORMANCE_CATEGORY,
    COL_PREDICTED_CATEGORY,
    COL_PREVIOUS_MARKS,
    COL_SLEEP_HOURS,
    COL_STUDENT_ID,
    COL_STUDY_HOURS,
    COLOR_ACCENT,
    COLOR_BG_LIGHT,
    COLOR_BORDER,
    COLOR_CARD_BG,
    COLOR_PRIMARY,
)


def inject_custom_styles():
    """Injects premium CSS styles into the Streamlit app."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* Global App Background */
    .stApp {{
        background-color: {COLOR_BG_LIGHT};
        color: #1E293B;
    }}

    /* Top Header Branding Banner */
    .brand-header-card {{
        background: linear-gradient(135deg, #0F5132 0%, #13653F 50%, #198754 100%);
        border-radius: 14px;
        padding: 24px 28px;
        color: #FFFFFF;
        box-shadow: 0 4px 20px -2px rgba(15, 81, 50, 0.25);
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    .brand-title {{
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .brand-subtitle {{
        font-size: 14px;
        color: #D1E7DD;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
    }}

    /* Modern Card Container */
    .ep-card {{
        background-color: {COLOR_CARD_BG};
        border-radius: 12px;
        border: 1px solid {COLOR_BORDER};
        padding: 20px 22px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .ep-card:hover {{
        box-shadow: 0 4px 12px -2px rgba(15, 81, 50, 0.08);
    }}

    /* KPI Metric Card */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: left;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: {COLOR_ACCENT};
    }}

    .kpi-label {{
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}

    .kpi-value {{
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }}

    .kpi-subtext {{
        font-size: 12px;
        color: #94A3B8;
        margin-top: 5px;
    }}

    /* Category Badges */
    .badge-at-risk {{
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}

    .badge-average {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}

    .badge-good {{
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {{
        background-color: #F8FAF9;
        border-right: 1px solid #E2E8F0;
    }}

    .sidebar-brand {{
        padding: 16px 8px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }}

    .sidebar-logo {{
        font-size: 20px;
        font-weight: 800;
        color: #0F5132;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Section Subheadings */
    .section-title {{
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .section-desc {{
        font-size: 13px;
        color: #64748B;
        margin-bottom: 18px;
    }}

    /* Info Callout Box */
    .callout-box {{
        background: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #166534;
        margin-bottom: 16px;
    }}

    .callout-box strong {{
        color: #0F5132;
    }}

    /* Button Styling */
    .stButton>button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    .stButton>button[kind="primary"] {{
        background-color: #0F5132;
        border-color: #0F5132;
        color: #FFFFFF;
    }}

    .stButton>button[kind="primary"]:hover {{
        background-color: #13653F;
        border-color: #13653F;
        color: #FFFFFF;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_brand_header(title: str, subtitle: str):
    """Renders the top branding card."""
    html = f"""
    <div class="brand-header-card">
        <h1 class="brand-title">
            <span>🎓</span> {title}
        </h1>
        <p class="brand-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", border_color: str = COLOR_ACCENT):
    """Renders a styled KPI metric card."""
    html = f"""
    <div class="kpi-card" style="border-top: 3px solid {border_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_category_badge(category: Optional[str]) -> str:
    """Returns HTML for category badge."""
    if category == CATEGORY_AT_RISK:
        return f'<span class="badge-at-risk">⚠️ {CATEGORY_AT_RISK}</span>'
    elif category == CATEGORY_AVERAGE:
        return f'<span class="badge-average">⚡ {CATEGORY_AVERAGE}</span>'
    elif category == CATEGORY_GOOD:
        return f'<span class="badge-good">🌟 {CATEGORY_GOOD}</span>'
    return '<span style="color:#94A3B8; font-style:italic;">Pending Prediction</span>'


# ==========================================
# PLOTLY CHART BUILDERS (Emerald Palette)
# ==========================================

def plot_category_donut(counts: pd.Series, title: str = "Student Performance Distribution") -> go.Figure:
    """Generates an interactive donut chart showing category distribution."""
    labels = list(counts.index)
    values = list(counts.values)
    colors = [CATEGORY_COLORS.get(label, "#94A3B8") for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13, family="Inter"),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter", color="#0F172A", weight=700)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_category_bar(counts: pd.Series, title: str = "Category Student Counts") -> go.Figure:
    """Generates a styled bar chart comparing category counts."""
    categories = [CATEGORY_AT_RISK, CATEGORY_AVERAGE, CATEGORY_GOOD]
    values = [counts.get(cat, 0) for cat in categories]
    colors = [CATEGORY_COLORS[cat] for cat in categories]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(0,0,0,0.1)", width=1),
                    cornerradius=6,
                ),
                text=values,
                textposition="auto",
                textfont=dict(size=14, family="Inter", color="#FFFFFF", weight=700),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter", color="#0F172A", weight=700)),
        xaxis=dict(title="", tickfont=dict(size=13, family="Inter", weight=600)),
        yaxis=dict(title="Student Count", gridcolor="#F1F5F9", rangemode="tozero"),
        margin=dict(l=30, r=20, t=50, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Generates an annotated heatmap of the holdout confusion matrix."""
    z = np.array(cm)
    total_samples = np.sum(z)

    # Format text annotations with counts and percentages
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            pct = (val / total_samples * 100) if total_samples > 0 else 0
            annotations.append(
                dict(
                    x=classes[j],
                    y=classes[i],
                    text=f"<b>{val}</b><br><span style='font-size:11px;'>({pct:.1f}%)</span>",
                    showarrow=False,
                    font=dict(
                        color="#FFFFFF" if val > (np.max(z) * 0.5) else "#0F172A",
                        size=14,
                        family="Inter",
                    ),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=classes,
            y=classes,
            colorscale=[[0, "#E8F5E9"], [0.5, "#4CAF50"], [1.0, "#0F5132"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="Holdout Evaluation Confusion Matrix",
            font=dict(size=15, family="Inter", color="#0F172A", weight=700),
        ),
        xaxis=dict(
            title="<b>Predicted Category</b>",
            tickfont=dict(size=12, family="Inter"),
        ),
        yaxis=dict(
            title="<b>Actual Category (Ground Truth)</b>",
            autorange="reversed",
            tickfont=dict(size=12, family="Inter"),
        ),
        annotations=annotations,
        margin=dict(l=40, r=30, t=50, b=40),
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_model_comparison(models_metrics: Dict[str, any]) -> go.Figure:
    """Generates grouped bar chart comparing baseline and candidate ML models."""
    names = list(models_metrics.keys())
    accuracies = [round(m.accuracy * 100, 1) for m in models_metrics.values()]
    f1_macros = [round(m.f1_macro * 100, 1) for m in models_metrics.values()]
    precisions = [round(m.precision_macro * 100, 1) for m in models_metrics.values()]
    recalls = [round(m.recall_macro * 100, 1) for m in models_metrics.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy (%)", x=names, y=accuracies, marker_color="#0F5132"))
    fig.add_trace(go.Bar(name="Macro F1 (%)", x=names, y=f1_macros, marker_color="#10B981"))
    fig.add_trace(go.Bar(name="Macro Precision (%)", x=names, y=precisions, marker_color="#34D399"))
    fig.add_trace(go.Bar(name="Macro Recall (%)", x=names, y=recalls, marker_color="#6EE7B7"))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Model Comparison on Held-Out Test Set",
            font=dict(size=15, family="Inter", color="#0F172A", weight=700),
        ),
        yaxis=dict(title="Score (%)", range=[0, 105], gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=30, r=20, t=50, b=50),
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_study_vs_marks(df: pd.DataFrame, category_col: str) -> go.Figure:
    """Scatter plot: Study Hours vs Previous Marks, colored by Category."""
    fig = px.scatter(
        df,
        x=COL_STUDY_HOURS,
        y=COL_PREVIOUS_MARKS,
        color=category_col,
        color_discrete_map=CATEGORY_COLORS,
        hover_data=[COL_STUDENT_ID, COL_ATTENDANCE, COL_ASSIGNMENT_AVG],
        labels={
            COL_STUDY_HOURS: "Daily Study Hours (hrs/day)",
            COL_PREVIOUS_MARKS: "Previous Academic Marks (%)",
            category_col: "Performance Category",
        },
        title="Study Hours vs. Previous Academic Marks",
    )

    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="#FFFFFF")))
    fig.update_layout(
        font=dict(family="Inter"),
        title=dict(font=dict(size=15, color="#0F172A", weight=700)),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9"),
        height=360,
        margin=dict(l=30, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_sleep_vs_attendance(df: pd.DataFrame, category_col: str) -> go.Figure:
    """Scatter plot: Sleep Hours vs Attendance Percentage."""
    fig = px.scatter(
        df,
        x=COL_SLEEP_HOURS,
        y=COL_ATTENDANCE,
        color=category_col,
        color_discrete_map=CATEGORY_COLORS,
        hover_data=[COL_STUDENT_ID, COL_INTERNAL_MARKS],
        labels={
            COL_SLEEP_HOURS: "Daily Sleep Hours (hrs/day)",
            COL_ATTENDANCE: "Attendance Rate (%)",
            category_col: "Performance Category",
        },
        title="Daily Sleep Hours vs. Class Attendance",
    )

    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="#FFFFFF")))
    fig.update_layout(
        font=dict(family="Inter"),
        title=dict(font=dict(size=15, color="#0F172A", weight=700)),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9"),
        height=360,
        margin=dict(l=30, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_feature_boxplots(df: pd.DataFrame, feature_col: str, feature_label: str, category_col: str) -> go.Figure:
    """Box plot of an academic feature across performance categories."""
    fig = px.box(
        df,
        x=category_col,
        y=feature_col,
        color=category_col,
        color_discrete_map=CATEGORY_COLORS,
        points="outliers",
        labels={
            category_col: "Category",
            feature_col: feature_label,
        },
        title=f"{feature_label} Distribution by Category",
    )

    fig.update_layout(
        font=dict(family="Inter"),
        title=dict(font=dict(size=15, color="#0F172A", weight=700)),
        showlegend=False,
        yaxis=dict(gridcolor="#F1F5F9"),
        height=340,
        margin=dict(l=30, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_feature_importance_bar(feature_importances: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing relative feature importances."""
    features = list(feature_importances.keys())[::-1]
    scores = [feature_importances[f] * 100 for f in features]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=features,
                orientation="h",
                marker=dict(
                    color="#10B981",
                    line=dict(color="#059669", width=1),
                    cornerradius=4,
                ),
                text=[f"{s:.1f}%" for s in scores],
                textposition="auto",
                textfont=dict(family="Inter", size=12, color="#064E3B", weight=600),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Model Feature Importance / Academic Predictive Weight",
            font=dict(size=15, family="Inter", color="#0F172A", weight=700),
        ),
        xaxis=dict(title="Importance Weight (%)", gridcolor="#F1F5F9"),
        yaxis=dict(title=""),
        margin=dict(l=150, r=30, t=50, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> go.Figure:
    """Generates correlation matrix heatmap for numeric features."""
    corr = df[feature_cols].corr()

    # Friendly labels
    from src.constants import FEATURE_PRETTY_NAMES
    labels = [FEATURE_PRETTY_NAMES.get(c, c) for c in feature_cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[[0, "#E2E8F0"], [0.5, "#A7F3D0"], [1.0, "#0F5132"]],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, family="Inter"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Academic Features Correlation Matrix",
            font=dict(size=15, family="Inter", color="#0F172A", weight=700),
        ),
        margin=dict(l=60, r=40, t=50, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
