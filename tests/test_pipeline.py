"""Tests for machine learning pipeline, holdout evaluation, model comparison, and prediction."""

import pandas as pd
import pytest

from src.constants import (
    CATEGORY_AT_RISK,
    CATEGORY_AVERAGE,
    CATEGORY_GOOD,
    COL_CONFIDENCE,
    COL_PREDICTED_CATEGORY,
    VALID_CATEGORIES,
)
from src.data_processor import validate_and_process_dataset
from src.ml_pipeline import (
    apply_predictions_to_full_dataset,
    generate_predictions,
    train_and_evaluate_pipeline,
)
from src.synthetic_data import generate_synthetic_students


def test_train_and_evaluate_success():
    """Verify training runs both baseline and RF models and computes holdout metrics."""
    df = generate_synthetic_students(num_labelled_per_class=20, num_unlabelled=5, random_seed=42)
    val_res = validate_and_process_dataset(df)

    pipeline_result = train_and_evaluate_pipeline(val_res.labelled_df, test_size=0.2)

    assert pipeline_result.best_model_name in [
        "Logistic Regression (Baseline)",
        "Random Forest Classifier",
    ]
    assert len(pipeline_result.models_metrics) == 2

    # Check metrics for both models
    for name, metrics in pipeline_result.models_metrics.items():
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1_macro <= 1.0
        assert 0.0 <= metrics.precision_macro <= 1.0
        assert 0.0 <= metrics.recall_macro <= 1.0
        assert len(metrics.classes) == 3
        assert len(metrics.confusion_matrix) == 3

    # Check feature importances
    assert len(pipeline_result.feature_importances) == 6
    assert sum(pipeline_result.feature_importances.values()) > 0


def test_prediction_generation():
    """Verify predictions and confidence scores for unlabelled rows."""
    df = generate_synthetic_students(num_labelled_per_class=20, num_unlabelled=10, random_seed=42)
    val_res = validate_and_process_dataset(df)
    pipeline_result = train_and_evaluate_pipeline(val_res.labelled_df)

    pred_df = generate_predictions(pipeline_result.best_model_pipeline, val_res.unlabelled_df)

    assert len(pred_df) == 10
    assert COL_PREDICTED_CATEGORY in pred_df.columns
    assert COL_CONFIDENCE in pred_df.columns

    for cat in pred_df[COL_PREDICTED_CATEGORY]:
        assert cat in VALID_CATEGORIES

    for conf in pred_df[COL_CONFIDENCE]:
        assert 0.33 <= conf <= 1.0


def test_apply_predictions_to_master():
    """Verify master dataframe preserves historical labels and populates unlabelled predictions."""
    df = generate_synthetic_students(num_labelled_per_class=15, num_unlabelled=8, random_seed=42)
    val_res = validate_and_process_dataset(df)
    pipeline_result = train_and_evaluate_pipeline(val_res.labelled_df)

    master_df = apply_predictions_to_full_dataset(
        pipeline_result.best_model_pipeline, val_res.cleaned_df
    )

    assert len(master_df) == len(df)
    assert not master_df[COL_PREDICTED_CATEGORY].isna().any()

    # Historical labels should be preserved
    labelled_original = val_res.labelled_df.copy()
    for _, row in labelled_original.iterrows():
        student_id = row["student_id"]
        master_row = master_df[master_df["student_id"] == student_id].iloc[0]
        assert master_row[COL_PREDICTED_CATEGORY] == row["performance_category"]


def test_insufficient_labelled_records():
    """Verify ValueError is raised when labelled records < 6."""
    df = generate_synthetic_students(num_labelled_per_class=1, num_unlabelled=0)  # Only 3 records
    val_res = validate_and_process_dataset(df)

    with pytest.raises(ValueError, match="Insufficient labelled historical data"):
        train_and_evaluate_pipeline(val_res.labelled_df)


def test_single_category_dataset():
    """Verify ValueError is raised when historical data only has 1 unique class."""
    df = generate_synthetic_students(num_labelled_per_class=10, num_unlabelled=0)
    # Set all categories to 'Good'
    df["performance_category"] = CATEGORY_GOOD
    val_res = validate_and_process_dataset(df)

    with pytest.raises(ValueError, match="only contains one category"):
        train_and_evaluate_pipeline(val_res.labelled_df)


def test_confusion_matrix_total_equals_test_size():
    """Verify that sum of elements in confusion matrix equals holdout test set size."""
    df = generate_synthetic_students(num_labelled_per_class=20, num_unlabelled=5, random_seed=42)
    val_res = validate_and_process_dataset(df)
    pipeline_res = train_and_evaluate_pipeline(val_res.labelled_df, test_size=0.25)

    for _, metrics in pipeline_res.models_metrics.items():
        total_cm_samples = sum(sum(row) for row in metrics.confusion_matrix)
        assert total_cm_samples == metrics.test_size


def test_prediction_on_empty_unlabelled():
    """Verify predict function handles empty unlabelled dataframe cleanly."""
    df = generate_synthetic_students(num_labelled_per_class=10, num_unlabelled=0, random_seed=42)
    val_res = validate_and_process_dataset(df)
    pipeline_res = train_and_evaluate_pipeline(val_res.labelled_df)

    pred_df = generate_predictions(pipeline_res.best_model_pipeline, pd.DataFrame(columns=df.columns))
    assert pred_df.empty
    assert COL_PREDICTED_CATEGORY in pred_df.columns
    assert COL_CONFIDENCE in pred_df.columns
