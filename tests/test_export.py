"""Tests for Excel and CSV export and template generation."""

import io
import openpyxl
import pandas as pd
import pytest

from src.constants import REQUIRED_COLUMNS
from src.data_processor import (
    export_predictions_to_csv,
    export_predictions_to_excel,
    generate_template_excel,
    validate_and_process_dataset,
)
from src.ml_pipeline import apply_predictions_to_full_dataset, train_and_evaluate_pipeline
from src.synthetic_data import generate_synthetic_students


def test_generate_template_excel():
    """Verify generated blank template contains correct required columns."""
    template_bytes = generate_template_excel()
    assert template_bytes is not None

    df_template = pd.read_excel(template_bytes, engine="openpyxl")
    for col in REQUIRED_COLUMNS:
        assert col in df_template.columns


def test_export_to_excel_and_reload():
    """Verify exporting predictions to Excel produces valid, readable file with intact data."""
    df = generate_synthetic_students(num_labelled_per_class=10, num_unlabelled=5, random_seed=42)
    val_res = validate_and_process_dataset(df)
    pipeline_res = train_and_evaluate_pipeline(val_res.labelled_df)
    master_df = apply_predictions_to_full_dataset(
        pipeline_res.best_model_pipeline, val_res.cleaned_df
    )

    excel_bytes = export_predictions_to_excel(master_df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Reload using pandas
    reloaded_df = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
    assert len(reloaded_df) == len(master_df)
    assert "student_id" in reloaded_df.columns
    assert "predicted_category" in reloaded_df.columns
    assert "confidence_score" in reloaded_df.columns


def test_export_to_csv():
    """Verify exporting predictions to CSV generates valid, readable CSV."""
    df = generate_synthetic_students(num_labelled_per_class=10, num_unlabelled=5, random_seed=42)
    val_res = validate_and_process_dataset(df)
    pipeline_res = train_and_evaluate_pipeline(val_res.labelled_df)
    master_df = apply_predictions_to_full_dataset(
        pipeline_res.best_model_pipeline, val_res.cleaned_df
    )

    csv_str = export_predictions_to_csv(master_df)
    assert isinstance(csv_str, str)
    assert len(csv_str) > 0

    reloaded_csv = pd.read_csv(io.StringIO(csv_str))
    assert len(reloaded_csv) == len(master_df)
    assert "predicted_category" in reloaded_csv.columns
