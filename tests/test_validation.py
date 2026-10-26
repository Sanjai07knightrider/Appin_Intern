"""Tests for data validation, schema enforcement, category normalization, and unlabelled row detection."""

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    CATEGORY_AT_RISK,
    CATEGORY_AVERAGE,
    CATEGORY_GOOD,
    COL_ASSIGNMENT_AVG,
    COL_ATTENDANCE,
    COL_INTERNAL_MARKS,
    COL_IS_PREDICTION,
    COL_PERFORMANCE_CATEGORY,
    COL_PREVIOUS_MARKS,
    COL_SLEEP_HOURS,
    COL_STUDENT_ID,
    COL_STUDY_HOURS,
    REQUIRED_COLUMNS,
)
from src.data_processor import validate_and_process_dataset
from src.synthetic_data import generate_synthetic_students


def test_valid_synthetic_data():
    """Verify that synthetic dataset passes validation completely."""
    df = generate_synthetic_students(num_labelled_per_class=10, num_unlabelled=5, random_seed=42)
    res = validate_and_process_dataset(df)

    assert res.is_valid is True
    assert len(res.errors) == 0
    assert len(res.labelled_df) == 30
    assert len(res.unlabelled_df) == 5
    assert len(res.cleaned_df) == 35


def test_missing_required_column():
    """Verify error when required column is missing."""
    df = generate_synthetic_students(num_labelled_per_class=5, num_unlabelled=2)
    df_missing = df.drop(columns=[COL_ATTENDANCE])

    res = validate_and_process_dataset(df_missing)
    assert res.is_valid is False
    assert any("attendance_pct" in err for err in res.errors)


def test_duplicate_student_ids():
    """Verify error when duplicate student IDs are present."""
    df = generate_synthetic_students(num_labelled_per_class=5, num_unlabelled=2)
    # Introduce duplicate ID
    df.loc[1, COL_STUDENT_ID] = df.loc[0, COL_STUDENT_ID]

    res = validate_and_process_dataset(df)
    assert res.is_valid is False
    assert any("Duplicate student_id" in err for err in res.errors)


def test_blank_student_id():
    """Verify error when student ID is blank or empty."""
    df = generate_synthetic_students(num_labelled_per_class=5, num_unlabelled=2)
    df.loc[2, COL_STUDENT_ID] = "   "

    res = validate_and_process_dataset(df)
    assert res.is_valid is False
    assert any("blank or invalid student_id" in err for err in res.errors)


def test_numeric_range_validation():
    """Verify validation flags values outside bounds [0-100] and [0-24]."""
    df = generate_synthetic_students(num_labelled_per_class=5, num_unlabelled=2)
    # Set attendance > 100
    df.loc[0, COL_ATTENDANCE] = 105.5
    # Set study hours > 24
    df.loc[1, COL_STUDY_HOURS] = 26.0

    res = validate_and_process_dataset(df)
    assert res.is_valid is False
    assert any("attendance_pct" in err and "[0.0, 100.0]" in err for err in res.errors)
    assert any("study_hours_per_day" in err and "[0.0, 24.0]" in err for err in res.errors)


def test_non_numeric_academic_values():
    """Verify validation detects non-numeric strings in numeric feature columns."""
    df = generate_synthetic_students(num_labelled_per_class=5, num_unlabelled=2)
    df[COL_INTERNAL_MARKS] = df[COL_INTERNAL_MARKS].astype(object)
    df.loc[0, COL_INTERNAL_MARKS] = "invalid_score"

    res = validate_and_process_dataset(df)
    assert res.is_valid is False
    assert any("internal_marks_pct" in err and "non-numeric" in err for err in res.errors)


def test_category_normalization():
    """Verify categories in varied casing and formats are properly normalized."""
    df = generate_synthetic_students(num_labelled_per_class=2, num_unlabelled=1)
    df.loc[0, COL_PERFORMANCE_CATEGORY] = "  at risk  "
    df.loc[1, COL_PERFORMANCE_CATEGORY] = "GOOD"
    df.loc[2, COL_PERFORMANCE_CATEGORY] = "average"

    res = validate_and_process_dataset(df)
    assert res.is_valid is True
    assert res.cleaned_df.loc[0, COL_PERFORMANCE_CATEGORY] == CATEGORY_AT_RISK
    assert res.cleaned_df.loc[1, COL_PERFORMANCE_CATEGORY] == CATEGORY_GOOD
    assert res.cleaned_df.loc[2, COL_PERFORMANCE_CATEGORY] == CATEGORY_AVERAGE


def test_invalid_category_rejected():
    """Verify unsupported categories are rejected with clear error."""
    df = generate_synthetic_students(num_labelled_per_class=2, num_unlabelled=1)
    df.loc[0, COL_PERFORMANCE_CATEGORY] = "Super Excellent"

    res = validate_and_process_dataset(df)
    assert res.is_valid is False
    assert any("Invalid performance_category labels" in err for err in res.errors)


def test_unlabelled_row_detection():
    """Verify empty/null/whitespace categories are treated as unlabelled prediction targets."""
    df = generate_synthetic_students(num_labelled_per_class=5, num_unlabelled=0)
    df.loc[0, COL_PERFORMANCE_CATEGORY] = None
    df.loc[1, COL_PERFORMANCE_CATEGORY] = np.nan
    df.loc[2, COL_PERFORMANCE_CATEGORY] = "   "

    res = validate_and_process_dataset(df)
    assert res.is_valid is True
    assert len(res.unlabelled_df) == 3
    assert len(res.labelled_df) == len(df) - 3
    assert bool(res.cleaned_df.loc[0, COL_IS_PREDICTION]) is True


def test_empty_dataframe():
    """Verify validation handles empty input gracefully without crashing."""
    res = validate_and_process_dataset(pd.DataFrame())
    assert res.is_valid is False
    assert any("empty" in err for err in res.errors)


def test_boundary_values():
    """Verify numeric boundaries [0, 100] and [0, 24] are accepted."""
    df = generate_synthetic_students(num_labelled_per_class=2, num_unlabelled=0)
    df.loc[0, COL_ATTENDANCE] = 0.0
    df.loc[0, COL_PREVIOUS_MARKS] = 100.0
    df.loc[0, COL_STUDY_HOURS] = 0.0
    df.loc[0, COL_SLEEP_HOURS] = 24.0

    res = validate_and_process_dataset(df)
    assert res.is_valid is True
    assert res.cleaned_df.loc[0, COL_ATTENDANCE] == 0.0
    assert res.cleaned_df.loc[0, COL_PREVIOUS_MARKS] == 100.0
    assert res.cleaned_df.loc[0, COL_STUDY_HOURS] == 0.0
    assert res.cleaned_df.loc[0, COL_SLEEP_HOURS] == 24.0

