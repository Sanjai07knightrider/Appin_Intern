"""Data validation, preprocessing, template generation, and export utilities."""

from dataclasses import dataclass, field
import io
from typing import List, Optional, Tuple
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.constants import (
    CATEGORY_NORMALIZATION_MAP,
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
    FEATURE_COLUMNS,
    NUMERIC_RANGES,
    REQUIRED_COLUMNS,
    VALID_CATEGORIES,
)


@dataclass
class ValidationResult:
    """Holds validation outputs, data subsets, and error/warning messages."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    labelled_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    unlabelled_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def validate_and_process_dataset(raw_df: pd.DataFrame) -> ValidationResult:
    """Validates the input DataFrame against EduPredict AI requirements.
    
    Checks:
    - Required column schema
    - Duplicate student IDs
    - Numeric data types and range validation [0-100 or 0-24]
    - Performance category normalization and detection of unlabelled rows
    - Minimum data size checks for training
    """
    errors: List[str] = []
    warnings: List[str] = []

    if raw_df is None or raw_df.empty:
        return ValidationResult(
            is_valid=False,
            errors=["Uploaded file is empty or contains no records."],
        )

    df = raw_df.copy()

    # 1. Normalize column names (strip whitespace and convert to lower case)
    df.columns = [str(col).strip().lower() for col in df.columns]

    # 2. Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required columns in uploaded workbook: {', '.join(missing_cols)}. "
            f"Expected schema: {', '.join(REQUIRED_COLUMNS)}."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # 3. Check for empty student IDs & duplicates
    # Convert student_id to string and strip
    df[COL_STUDENT_ID] = df[COL_STUDENT_ID].astype(str).str.strip()
    blank_ids = df[df[COL_STUDENT_ID].isin(["", "nan", "none", "null"])]
    if not blank_ids.empty:
        errors.append(f"Found {len(blank_ids)} row(s) with blank or invalid student_id.")

    duplicates = df[df.duplicated(subset=[COL_STUDENT_ID], keep=False)]
    if not duplicates.empty:
        dup_ids = duplicates[COL_STUDENT_ID].unique().tolist()[:5]
        suffix = "..." if len(duplicates[COL_STUDENT_ID].unique()) > 5 else ""
        errors.append(
            f"Duplicate student_id detected for {len(duplicates[COL_STUDENT_ID].unique())} ID(s): "
            f"{', '.join(dup_ids)}{suffix}. Student IDs must be strictly unique."
        )

    # 4. Check numeric features and ranges
    for col in FEATURE_COLUMNS:
        # Check if numeric
        original_vals = df[col]
        numeric_series = pd.to_numeric(original_vals, errors="coerce")
        null_count = numeric_series.isna().sum()

        if null_count > 0:
            errors.append(
                f"Column '{col}' has {null_count} non-numeric or missing value(s). "
                "All academic feature values must be valid numbers."
            )
            continue

        df[col] = numeric_series

        min_val, max_val = NUMERIC_RANGES[col]
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            sample_ids = out_of_bounds[COL_STUDENT_ID].iloc[:3].tolist()
            errors.append(
                f"Column '{col}' contains {len(out_of_bounds)} value(s) out of valid range [{min_val}, {max_val}]. "
                f"Example student IDs: {', '.join(sample_ids)}."
            )

    # 5. Process performance_category column
    # Distinguish labelled historical rows from unlabelled prediction target rows
    is_unlabelled = df[COL_PERFORMANCE_CATEGORY].isna() | (
        df[COL_PERFORMANCE_CATEGORY].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "?"])
    )

    normalized_categories = []
    invalid_category_errors = []

    for idx, (val, unlabelled) in enumerate(zip(df[COL_PERFORMANCE_CATEGORY], is_unlabelled)):
        if unlabelled:
            normalized_categories.append(None)
        else:
            val_str = str(val).strip().lower()
            if val_str in CATEGORY_NORMALIZATION_MAP:
                normalized_categories.append(CATEGORY_NORMALIZATION_MAP[val_str])
            else:
                student = df[COL_STUDENT_ID].iloc[idx]
                invalid_category_errors.append(f"Row {idx+1} (Student {student}): '{val}'")

    if invalid_category_errors:
        errors.append(
            f"Invalid performance_category labels found in {len(invalid_category_errors)} row(s): "
            f"{', '.join(invalid_category_errors[:5])}. Allowed labels: {', '.join(VALID_CATEGORIES)} (or blank for prediction)."
        )

    if errors:
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    df[COL_PERFORMANCE_CATEGORY] = normalized_categories
    df[COL_IS_PREDICTION] = is_unlabelled

    labelled_df = df[~is_unlabelled].copy().reset_index(drop=True)
    unlabelled_df = df[is_unlabelled].copy().reset_index(drop=True)

    # Check minimum dataset guidelines
    if len(labelled_df) == 0:
        warnings.append(
            "Uploaded file contains 0 labelled historical records. The model requires historical labels "
            "to train and evaluate before it can predict unlabelled rows."
        )
    elif len(labelled_df) < 15:
        warnings.append(
            f"Only {len(labelled_df)} labelled historical rows provided. For robust cross-validation "
            "and stratified holdout evaluation, at least 15–30 labelled records are recommended."
        )

    # Check class coverage in labelled data
    if len(labelled_df) > 0:
        present_classes = set(labelled_df[COL_PERFORMANCE_CATEGORY].dropna().unique())
        missing_classes = set(VALID_CATEGORIES) - present_classes
        if missing_classes:
            warnings.append(
                f"Historical labelled data is missing sample(s) for category: {', '.join(missing_classes)}. "
                "The model will only be able to learn patterns for categories present in training data."
            )

    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=warnings,
        labelled_df=labelled_df,
        unlabelled_df=unlabelled_df,
        cleaned_df=df,
    )


def generate_template_excel() -> io.BytesIO:
    """Generates a styled, blank sample template Excel file for users to download."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Student_Data_Template"

    # Header styling
    header_fill = PatternFill(start_color="0F5132", end_color="0F5132", fill_type="solid")
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin", color="CCCCCC"),
        right=Side(style="thin", color="CCCCCC"),
        top=Side(style="thin", color="CCCCCC"),
        bottom=Side(style="thin", color="CCCCCC"),
    )

    headers = [
        "student_id",
        "attendance_pct",
        "previous_marks_pct",
        "assignment_avg_pct",
        "internal_marks_pct",
        "study_hours_per_day",
        "sleep_hours_per_day",
        "performance_category",
    ]

    ws.append(headers)

    for col_num in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Add 2 sample guide rows (1 labelled, 1 blank for prediction)
    sample_rows = [
        ["STU-EXAMPLE-01", 88.5, 82.0, 85.0, 80.0, 5.0, 7.5, "Good"],
        ["STU-EXAMPLE-02", 52.0, 48.0, 50.0, 45.0, 1.5, 5.0, "At Risk"],
        ["STU-EXAMPLE-03", 72.0, 68.0, 70.0, 66.0, 3.5, 7.0, ""],  # Blank for prediction
    ]

    guide_fill = PatternFill(start_color="F0FDF4", end_color="F0FDF4", fill_type="solid")
    guide_font = Font(name="Arial", size=10, italic=True, color="333333")

    for row_idx, row_data in enumerate(sample_rows, start=2):
        ws.append(row_data)
        for col_num in range(1, len(row_data) + 1):
            cell = ws.cell(row=row_idx, column=col_num)
            cell.fill = guide_fill
            cell.font = guide_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")

    # Auto-adjust column widths
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 16)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


def export_predictions_to_excel(df: pd.DataFrame) -> bytes:
    """Exports dataset containing predictions to an elegant, formatted Excel workbook."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "EduPredict_Results"

    # Styling definitions
    header_fill = PatternFill(start_color="0F5132", end_color="0F5132", fill_type="solid")
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style="thin", color="E2E8F0"),
        right=Side(style="thin", color="E2E8F0"),
        top=Side(style="thin", color="E2E8F0"),
        bottom=Side(style="thin", color="E2E8F0"),
    )

    # Category fills
    fill_at_risk = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    fill_average = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    fill_good = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")

    cols = list(df.columns)
    ws.append(cols)

    for col_idx in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    # Append data rows
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        row_list = list(row)
        ws.append(row_list)
        for col_idx, val in enumerate(row_list, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = thin_border
            cell.alignment = center_align

            # Colorize predicted_category or performance_category if relevant
            col_name = cols[col_idx - 1]
            if col_name in [COL_PREDICTED_CATEGORY, COL_PERFORMANCE_CATEGORY]:
                if val == "At Risk":
                    cell.fill = fill_at_risk
                    cell.font = Font(bold=True, color="991B1B")
                elif val == "Average":
                    cell.fill = fill_average
                    cell.font = Font(bold=True, color="92400E")
                elif val == "Good":
                    cell.fill = fill_good
                    cell.font = Font(bold=True, color="065F46")

            # Format confidence
            if col_name == COL_CONFIDENCE and isinstance(val, (int, float)):
                cell.number_format = "0.0%"

    # Auto-adjust widths
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 15)

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def export_predictions_to_csv(df: pd.DataFrame) -> str:
    """Exports prediction dataframe to CSV formatted string."""
    return df.to_csv(index=False)
