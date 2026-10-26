"""EduPredict AI — Premium Student Performance Prediction Web Application.

Full-stack Streamlit app featuring data validation, real ML training and holdout evaluation,
interactive exploration, and multi-format export.
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    CATEGORY_AT_RISK,
    CATEGORY_AVERAGE,
    CATEGORY_COLORS,
    CATEGORY_GOOD,
    COL_ASSIGNMENT_AVG,
    COL_ATTENDANCE,
    COL_CONFIDENCE,
    COL_INTERNAL_MARKS,
    COL_IS_PREDICTION,
    COL_PERFORMANCE_CATEGORY,
    COL_PREDICTED_CATEGORY,
    COL_PREVIOUS_MARKS,
    COL_SLEEP_HOURS,
    COL_STUDENT_ID,
    COL_STUDY_HOURS,
    COLOR_ACCENT,
    FEATURE_COLUMNS,
    FEATURE_PRETTY_NAMES,
    REQUIRED_COLUMNS,
    VALID_CATEGORIES,
)
from src.data_processor import (
    ValidationResult,
    export_predictions_to_csv,
    export_predictions_to_excel,
    generate_template_excel,
    validate_and_process_dataset,
)
from src.ml_pipeline import (
    PipelineTrainingResult,
    apply_predictions_to_full_dataset,
    train_and_evaluate_pipeline,
)
from src.synthetic_data import generate_synthetic_students
from src.ui_components import (
    inject_custom_styles,
    plot_category_bar,
    plot_category_donut,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_feature_importance_bar,
    plot_model_comparison,
    plot_sleep_vs_attendance,
    plot_study_vs_marks,
    render_brand_header,
    render_category_badge,
    render_kpi_card,
)

# Page configuration
st.set_page_config(
    page_title="EduPredict AI — Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom typography and emerald styling
inject_custom_styles()


# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
def initialize_session_state():
    """Initializes session state variables."""
    if "current_df" not in st.session_state:
        # Preload synthetic sample dataset for immediate demonstration
        synthetic_df = generate_synthetic_students(
            num_labelled_per_class=30, num_unlabelled=20, random_seed=42
        )
        validation_res = validate_and_process_dataset(synthetic_df)
        st.session_state["raw_df"] = synthetic_df
        st.session_state["validation_result"] = validation_res
        st.session_state["source_type"] = "sample"

        # Auto-train initial model on sample data
        try:
            model_results = train_and_evaluate_pipeline(validation_res.labelled_df)
            master_df = apply_predictions_to_full_dataset(
                model_results.best_model_pipeline, validation_res.cleaned_df
            )
            st.session_state["model_results"] = model_results
            st.session_state["master_df"] = master_df
            st.session_state["is_trained"] = True
        except Exception as e:
            st.session_state["model_results"] = None
            st.session_state["master_df"] = validation_res.cleaned_df
            st.session_state["is_trained"] = False
            st.session_state["train_error"] = str(e)


initialize_session_state()


# -------------------------------------------------------------
# Sidebar Navigation and Data Management
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🎓 EduPredict AI</div>
            <div style="font-size: 11px; color: #64748B; margin-top: 2px;">
                v2.0 • Intelligent Academic Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Persistent Navigation Menu
    nav_view = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Student Explorer", "📈 Performance Insights"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin: 18px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)
    st.markdown("### 📁 Data Source")

    data_mode = st.radio(
        "Select Data Source",
        ["Use Realistic Sample Dataset", "Upload Custom Excel (.xlsx)"],
        index=0 if st.session_state.get("source_type") == "sample" else 1,
        label_visibility="collapsed",
    )

    if data_mode == "Upload Custom Excel (.xlsx)":
        uploaded_file = st.file_uploader(
            "Upload Excel Workbook",
            type=["xlsx"],
            help="Upload an .xlsx file containing required student academic columns.",
        )
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_excel(uploaded_file, engine="openpyxl")
                val_res = validate_and_process_dataset(uploaded_df)
                st.session_state["raw_df"] = uploaded_df
                st.session_state["validation_result"] = val_res
                st.session_state["source_type"] = "uploaded"

                if val_res.is_valid and len(val_res.labelled_df) >= 6:
                    model_res = train_and_evaluate_pipeline(val_res.labelled_df)
                    master_df = apply_predictions_to_full_dataset(
                        model_res.best_model_pipeline, val_res.cleaned_df
                    )
                    st.session_state["model_results"] = model_res
                    st.session_state["master_df"] = master_df
                    st.session_state["is_trained"] = True
                else:
                    st.session_state["model_results"] = None
                    st.session_state["master_df"] = val_res.cleaned_df
                    st.session_state["is_trained"] = False
            except Exception as exc:
                st.error(f"Error parsing uploaded Excel: {exc}")
    else:
        if st.session_state.get("source_type") != "sample":
            synthetic_df = generate_synthetic_students(
                num_labelled_per_class=30, num_unlabelled=20, random_seed=42
            )
            val_res = validate_and_process_dataset(synthetic_df)
            st.session_state["raw_df"] = synthetic_df
            st.session_state["validation_result"] = val_res
            st.session_state["source_type"] = "sample"
            model_res = train_and_evaluate_pipeline(val_res.labelled_df)
            st.session_state["model_results"] = model_res
            st.session_state["master_df"] = apply_predictions_to_full_dataset(
                model_res.best_model_pipeline, val_res.cleaned_df
            )
            st.session_state["is_trained"] = True
            st.rerun()

    # Template Download Button
    st.markdown("<hr style='margin: 16px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)
    st.markdown("#### 📥 Excel Template")
    template_bytes = generate_template_excel()
    st.download_button(
        label="Download Clean Template (.xlsx)",
        data=template_bytes,
        file_name="EduPredict_Student_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="font-size: 11px; color: #64748B; margin-top: 18px; line-height: 1.4;">
            <strong>Integrity Notice:</strong> EduPredict AI evaluates real ML models on stratified holdout test sets. Predictions are decision-support estimates to guide academic intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# Data & Validation Status Check
# -------------------------------------------------------------
val_result: ValidationResult = st.session_state.get("validation_result")
model_result: Optional[PipelineTrainingResult] = st.session_state.get("model_results")
master_df: Optional[pd.DataFrame] = st.session_state.get("master_df")

if not val_result or not val_result.is_valid:
    render_brand_header(
        "EduPredict AI — Data Validation",
        "Please address the data schema or range errors below before training models.",
    )
    st.error("### ⚠️ Excel Data Validation Failed")
    if val_result and val_result.errors:
        for err in val_result.errors:
            st.markdown(f"- **Error:** {err}")
    st.info(
        "Please download the provided **Clean Template (.xlsx)** from the left sidebar to see the required column schema and format."
    )
    st.stop()

# Display any warnings
if val_result.warnings:
    with st.expander("🔔 Dataset Notices & Recommendations", expanded=False):
        for warn in val_result.warnings:
            st.warning(warn)


# -------------------------------------------------------------
# VIEW 1: OVERVIEW
# -------------------------------------------------------------
if nav_view == "📊 Overview":
    render_brand_header(
        "EduPredict AI — Executive Overview",
        "Real-time student cohort analytics, verified ML performance metrics, and prediction pipeline.",
    )

    # 1. KPI Metrics Row
    total_students = len(master_df) if master_df is not None else 0
    labelled_count = len(val_result.labelled_df)
    unlabelled_count = len(val_result.unlabelled_df)

    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

    with col_kpi1:
        render_kpi_card("Total Students", f"{total_students:,}", "Processed in workbook", "#0F5132")
    with col_kpi2:
        render_kpi_card("Labelled History", f"{labelled_count:,}", "Used for model training", "#10B981")
    with col_kpi3:
        render_kpi_card("Predictions Needed", f"{unlabelled_count:,}", "Target unlabelled rows", "#3B82F6")
    with col_kpi4:
        if model_result:
            best_metrics = model_result.models_metrics[model_result.best_model_name]
            acc_str = f"{best_metrics.accuracy * 100:.1f}%"
            render_kpi_card("Holdout Accuracy", acc_str, f"Tested on {model_result.test_size} students", "#059669")
        else:
            render_kpi_card("Holdout Accuracy", "N/A", "Insufficient training rows", "#94A3B8")
    with col_kpi5:
        if model_result:
            best_metrics = model_result.models_metrics[model_result.best_model_name]
            f1_str = f"{best_metrics.f1_macro * 100:.1f}%"
            render_kpi_card("Holdout Macro F1", f1_str, "Balanced across all 3 classes", "#6366F1")
        else:
            render_kpi_card("Holdout Macro F1", "N/A", "Insufficient training rows", "#94A3B8")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # 2. Charts Row: Category Distribution Donut & Bar
    if master_df is not None and not master_df.empty:
        col_c1, col_c2 = st.columns(2)
        category_counts = master_df[COL_PREDICTED_CATEGORY].value_counts()

        with col_c1:
            st.markdown('<div class="ep-card">', unsafe_allow_html=True)
            donut_fig = plot_category_donut(category_counts, "Overall Performance Category Distribution")
            st.plotly_chart(donut_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c2:
            st.markdown('<div class="ep-card">', unsafe_allow_html=True)
            bar_fig = plot_category_bar(category_counts, "Student Headcount by Category")
            st.plotly_chart(bar_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # 3. Real Machine Learning Evaluation Panel
    if model_result:
        st.markdown(
            """
            <div class="section-title">
                <span>🔬</span> Verified Machine Learning Pipeline & Holdout Evaluation
            </div>
            <div class="section-desc">
                Models were trained strictly on training partitions and evaluated on a held-out test split (20%). 
                Final deployable model was retrained on all valid labelled historical rows.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown('<div class="ep-card">', unsafe_allow_html=True)
            comp_fig = plot_model_comparison(model_result.models_metrics)
            st.plotly_chart(comp_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown('<div class="ep-card">', unsafe_allow_html=True)
            best_m = model_result.models_metrics[model_result.best_model_name]
            cm_fig = plot_confusion_matrix(best_m.confusion_matrix, best_m.classes)
            st.plotly_chart(cm_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Classification Report Table
        with st.expander("📋 View Detailed Per-Class Holdout Classification Report", expanded=False):
            report_dict = best_m.classification_report_dict
            report_rows = []
            for cls in best_m.classes:
                if cls in report_dict:
                    report_rows.append({
                        "Category": cls,
                        "Precision": f"{report_dict[cls]['precision'] * 100:.1f}%",
                        "Recall": f"{report_dict[cls]['recall'] * 100:.1f}%",
                        "F1-Score": f"{report_dict[cls]['f1-score'] * 100:.1f}%",
                        "Test Support": int(report_dict[cls]["support"]),
                    })
            report_df = pd.DataFrame(report_rows)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

    # 4. Recent Predictions Summary
    if master_df is not None:
        unlabelled_only = master_df[master_df[COL_IS_PREDICTION] == True]
        if not unlabelled_only.empty:
            st.markdown(
                """
                <div class="section-title">
                    <span>⚡</span> Recent Model Predictions (Students Requiring Category Assignment)
                </div>
                <div class="section-desc">
                    Showing students who had blank historical categories in the workbook and were predicted by the trained ML pipeline.
                </div>
                """,
                unsafe_allow_html=True,
            )

            preview_cols = [
                COL_STUDENT_ID,
                COL_ATTENDANCE,
                COL_PREVIOUS_MARKS,
                COL_ASSIGNMENT_AVG,
                COL_INTERNAL_MARKS,
                COL_STUDY_HOURS,
                COL_SLEEP_HOURS,
                COL_PREDICTED_CATEGORY,
                COL_CONFIDENCE,
            ]
            display_preview = unlabelled_only[preview_cols].head(8).copy()
            if COL_CONFIDENCE in display_preview.columns:
                display_preview[COL_CONFIDENCE] = display_preview[COL_CONFIDENCE].apply(
                    lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "—"
                )

            st.dataframe(
                display_preview.rename(
                    columns={
                        COL_STUDENT_ID: "Student ID",
                        COL_ATTENDANCE: "Attendance (%)",
                        COL_PREVIOUS_MARKS: "Prev Marks (%)",
                        COL_ASSIGNMENT_AVG: "Assignments (%)",
                        COL_INTERNAL_MARKS: "Internals (%)",
                        COL_STUDY_HOURS: "Study Hrs/Day",
                        COL_SLEEP_HOURS: "Sleep Hrs/Day",
                        COL_PREDICTED_CATEGORY: "Predicted Category",
                        COL_CONFIDENCE: "Model Confidence",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# -------------------------------------------------------------
# VIEW 2: STUDENT EXPLORER
# -------------------------------------------------------------
elif nav_view == "🔍 Student Explorer":
    render_brand_header(
        "EduPredict AI — Student Explorer",
        "Search, filter, inspect individual academic scorecards, and export complete prediction workbooks.",
    )

    if master_df is None or master_df.empty:
        st.warning("No student records available to explore.")
        st.stop()

    # Search & Filter Toolbar
    col_search, col_filter, col_type = st.columns([2, 1.5, 1.5])

    with col_search:
        search_query = st.text_input(
            "Search by Student ID",
            placeholder="e.g. STU-1005",
            label_visibility="visible",
        ).strip().lower()

    with col_filter:
        category_options = ["All Categories", CATEGORY_AT_RISK, CATEGORY_AVERAGE, CATEGORY_GOOD]
        selected_category = st.selectbox("Filter by Category", category_options, index=0)

    with col_type:
        type_options = ["All Records", "Predictions Only (Unlabelled)", "Historical Labelled Records"]
        selected_type = st.selectbox("Record Type", type_options, index=0)

    # Apply filters
    filtered_df = master_df.copy()

    if search_query:
        filtered_df = filtered_df[
            filtered_df[COL_STUDENT_ID].astype(str).str.lower().str.contains(search_query)
        ]

    if selected_category != "All Categories":
        filtered_df = filtered_df[filtered_df[COL_PREDICTED_CATEGORY] == selected_category]

    if selected_type == "Predictions Only (Unlabelled)":
        filtered_df = filtered_df[filtered_df[COL_IS_PREDICTION] == True]
    elif selected_type == "Historical Labelled Records":
        filtered_df = filtered_df[filtered_df[COL_IS_PREDICTION] == False]

    # Filtered Results Stats Banner
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:10px 16px; margin: 12px 0 16px 0; font-size:13px; color:#475569;">
            Found <strong>{len(filtered_df)}</strong> matching student record(s) out of <strong>{len(master_df)}</strong> total students.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data Table View
    display_df = filtered_df[
        [
            COL_STUDENT_ID,
            COL_ATTENDANCE,
            COL_PREVIOUS_MARKS,
            COL_ASSIGNMENT_AVG,
            COL_INTERNAL_MARKS,
            COL_STUDY_HOURS,
            COL_SLEEP_HOURS,
            COL_PERFORMANCE_CATEGORY,
            COL_PREDICTED_CATEGORY,
            COL_CONFIDENCE,
        ]
    ].copy()

    display_df[COL_CONFIDENCE] = display_df[COL_CONFIDENCE].apply(
        lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "—"
    )
    display_df[COL_PERFORMANCE_CATEGORY] = display_df[COL_PERFORMANCE_CATEGORY].fillna("—")

    st.dataframe(
        display_df.rename(
            columns={
                COL_STUDENT_ID: "Student ID",
                COL_ATTENDANCE: "Attendance (%)",
                COL_PREVIOUS_MARKS: "Prev Marks (%)",
                COL_ASSIGNMENT_AVG: "Assignments (%)",
                COL_INTERNAL_MARKS: "Internals (%)",
                COL_STUDY_HOURS: "Study (hrs)",
                COL_SLEEP_HOURS: "Sleep (hrs)",
                COL_PERFORMANCE_CATEGORY: "Historical Label",
                COL_PREDICTED_CATEGORY: "Predicted Category",
                COL_CONFIDENCE: "Confidence",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Export Buttons
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        excel_bytes = export_predictions_to_excel(filtered_df)
        st.download_button(
            label="📥 Download Results as Excel (.xlsx)",
            data=excel_bytes,
            file_name="EduPredict_Student_Predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_exp2:
        csv_str = export_predictions_to_csv(filtered_df)
        st.download_button(
            label="📥 Download Results as CSV (.csv)",
            data=csv_str,
            file_name="EduPredict_Student_Predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<hr style='margin: 24px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # Deep-Dive Individual Student Detail Panel
    st.markdown(
        """
        <div class="section-title">
            <span>👤</span> Individual Student Academic Scorecard & Risk Diagnosis
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_student_ids = master_df[COL_STUDENT_ID].tolist()
    default_id = filtered_df[COL_STUDENT_ID].iloc[0] if not filtered_df.empty else all_student_ids[0]

    chosen_student_id = st.selectbox(
        "Select Student ID to Inspect",
        all_student_ids,
        index=all_student_ids.index(default_id) if default_id in all_student_ids else 0,
    )

    student_row = master_df[master_df[COL_STUDENT_ID] == chosen_student_id].iloc[0]

    # Render Student Card
    col_d1, col_d2, col_d3 = st.columns([1.2, 1.2, 1.6])

    with col_d1:
        st.markdown(
            f"""
            <div class="ep-card">
                <div style="font-size:12px; color:#64748B; font-weight:600; text-transform:uppercase;">Student Profile</div>
                <div style="font-size:24px; font-weight:800; color:#0F172A; margin:4px 0 12px 0;">{chosen_student_id}</div>
                <div style="margin-bottom:8px;">
                    <span style="font-size:13px; color:#64748B;">Category: </span>
                    {render_category_badge(student_row[COL_PREDICTED_CATEGORY])}
                </div>
                <div style="margin-bottom:8px;">
                    <span style="font-size:13px; color:#64748B;">Origin: </span>
                    <strong>{'Prediction' if student_row[COL_IS_PREDICTION] else 'Historical Record'}</strong>
                </div>
                <div>
                    <span style="font-size:13px; color:#64748B;">Model Confidence: </span>
                    <strong>{student_row[COL_CONFIDENCE]*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_d2:
        st.markdown(
            f"""
            <div class="ep-card">
                <div style="font-size:12px; color:#64748B; font-weight:600; text-transform:uppercase;">Academic Metrics</div>
                <div style="display:flex; justify-content:space-between; margin-top:8px;">
                    <span style="color:#64748B;">Attendance:</span>
                    <strong>{student_row[COL_ATTENDANCE]:.1f}%</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                    <span style="color:#64748B;">Previous Marks:</span>
                    <strong>{student_row[COL_PREVIOUS_MARKS]:.1f}%</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                    <span style="color:#64748B;">Assignments:</span>
                    <strong>{student_row[COL_ASSIGNMENT_AVG]:.1f}%</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                    <span style="color:#64748B;">Internals:</span>
                    <strong>{student_row[COL_INTERNAL_MARKS]:.1f}%</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                    <span style="color:#64748B;">Daily Study:</span>
                    <strong>{student_row[COL_STUDY_HOURS]:.1f} hrs/day</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                    <span style="color:#64748B;">Daily Sleep:</span>
                    <strong>{student_row[COL_SLEEP_HOURS]:.1f} hrs/day</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_d3:
        # Academic Diagnosis and Recommendations
        pred_cat = student_row[COL_PREDICTED_CATEGORY]
        risk_factors = []

        if student_row[COL_ATTENDANCE] < 70:
            risk_factors.append(f"Low Attendance ({student_row[COL_ATTENDANCE]}%)")
        if student_row[COL_STUDY_HOURS] < 2.5:
            risk_factors.append(f"Low Daily Study Time ({student_row[COL_STUDY_HOURS]} hrs)")
        if student_row[COL_INTERNAL_MARKS] < 55:
            risk_factors.append(f"Weak Internal Exam Scores ({student_row[COL_INTERNAL_MARKS]}%)")
        if student_row[COL_SLEEP_HOURS] < 6.0:
            risk_factors.append(f"Insufficient Sleep ({student_row[COL_SLEEP_HOURS]} hrs)")

        if pred_cat == CATEGORY_AT_RISK:
            recommendation = (
                "Immediate academic intervention required. Schedule 1-on-1 faculty mentoring, "
                "enroll in remedial study sessions, and monitor attendance weekly."
            )
        elif pred_cat == CATEGORY_AVERAGE:
            recommendation = (
                "Moderate performance trajectory. Suggest structured assignment tutoring "
                "and setting target milestones for internal examinations."
            )
        else:
            recommendation = (
                "Strong performance trajectory. Recommend advanced coursework, honors projects, "
                "or peer-tutoring mentorship roles."
            )

        risks_html = "".join([f"<li>{r}</li>" for r in risk_factors]) if risk_factors else "<li>No critical risk flags detected</li>"

        st.markdown(
            f"""
            <div class="ep-card">
                <div style="font-size:12px; color:#64748B; font-weight:600; text-transform:uppercase;">Advisor Recommendations</div>
                <p style="font-size:13px; color:#1E293B; margin:8px 0;">{recommendation}</p>
                <div style="font-size:12px; font-weight:600; color:#64748B; text-transform:uppercase; margin-top:8px;">Identified Risk Indicators:</div>
                <ul style="font-size:12px; color:#B91C1C; margin:4px 0 0 16px; padding:0;">
                    {risks_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------
# VIEW 3: PERFORMANCE INSIGHTS
# -------------------------------------------------------------
elif nav_view == "📈 Performance Insights":
    render_brand_header(
        "EduPredict AI — Academic Performance Insights",
        "Empirical feature relationships, correlation structures, and machine learning importance weights.",
    )

    if master_df is None or master_df.empty:
        st.warning("No dataset loaded for performance insights.")
        st.stop()

    # Academic Disclaimer Callout
    st.markdown(
        """
        <div class="callout-box">
            <strong>Methodology Note:</strong> The visualizations below depict empirical patterns observed in your student dataset. 
            Remember that statistical correlations indicate associations, not direct causation. These insights serve as decision-support signals 
            for academic counseling and pedagogical intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Study vs Marks & Sleep vs Attendance
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown('<div class="ep-card">', unsafe_allow_html=True)
        fig_study_marks = plot_study_vs_marks(master_df, COL_PREDICTED_CATEGORY)
        st.plotly_chart(fig_study_marks, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_i2:
        st.markdown('<div class="ep-card">', unsafe_allow_html=True)
        fig_sleep_att = plot_sleep_vs_attendance(master_df, COL_PREDICTED_CATEGORY)
        st.plotly_chart(fig_sleep_att, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Feature Distribution Boxplot & Feature Importance Bar
    col_i3, col_i4 = st.columns(2)

    with col_i3:
        st.markdown('<div class="ep-card">', unsafe_allow_html=True)
        feature_choice = st.selectbox(
            "Select Metric for Category Boxplot",
            options=FEATURE_COLUMNS,
            format_func=lambda x: FEATURE_PRETTY_NAMES.get(x, x),
            index=0,
        )
        fig_box = plot_feature_boxplots(
            master_df,
            feature_choice,
            FEATURE_PRETTY_NAMES.get(feature_choice, feature_choice),
            COL_PREDICTED_CATEGORY,
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_i4:
        st.markdown('<div class="ep-card">', unsafe_allow_html=True)
        if model_result and model_result.feature_importances:
            fig_imp = plot_feature_importance_bar(model_result.feature_importances)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance will appear once the machine learning model is trained.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Full Feature Correlation Heatmap
    st.markdown('<div class="ep-card">', unsafe_allow_html=True)
    fig_corr = plot_correlation_heatmap(master_df, FEATURE_COLUMNS)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
