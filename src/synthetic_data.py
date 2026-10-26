"""Synthetic dataset generator for EduPredict AI demonstration.

Generates realistic, synthetic student academic profiles across three performance categories
with realistic noise and overlap, including both historical labelled records and unlabelled
prediction rows.
"""

import os
from typing import Tuple
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.constants import (
    CATEGORY_AT_RISK,
    CATEGORY_AVERAGE,
    CATEGORY_GOOD,
    COL_ASSIGNMENT_AVG,
    COL_ATTENDANCE,
    COL_INTERNAL_MARKS,
    COL_PERFORMANCE_CATEGORY,
    COL_PREVIOUS_MARKS,
    COL_SLEEP_HOURS,
    COL_STUDENT_ID,
    COL_STUDY_HOURS,
    REQUIRED_COLUMNS,
)


def generate_synthetic_students(
    num_labelled_per_class: int = 30,
    num_unlabelled: int = 25,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic synthetic student academic records.
    
    Academic characteristics per category:
    - Good:
        attendance: 82 - 98%, previous marks: 78 - 96%, assignments: 80 - 95%,
        internals: 76 - 95%, study hours: 4.0 - 7.5 hrs, sleep: 6.5 - 8.5 hrs
    - Average:
        attendance: 65 - 85%, previous marks: 55 - 75%, assignments: 60 - 78%,
        internals: 55 - 75%, study hours: 2.5 - 4.5 hrs, sleep: 6.0 - 8.0 hrs
    - At Risk:
        attendance: 40 - 68%, previous marks: 32 - 55%, assignments: 35 - 58%,
        internals: 30 - 52%, study hours: 0.8 - 2.8 hrs, sleep: 4.5 - 6.5 hrs
    """
    rng = np.random.default_rng(random_seed)
    records = []
    id_counter = 1001

    # Define distributions for labelled data: (mean, std, min_clip, max_clip)
    profiles = {
        CATEGORY_GOOD: {
            "attendance": (88.0, 5.0, 75.0, 99.0),
            "prev_marks": (85.0, 6.0, 72.0, 99.0),
            "assignment": (87.0, 5.0, 74.0, 99.0),
            "internals": (84.0, 6.0, 70.0, 98.0),
            "study_hrs": (5.5, 1.0, 3.5, 9.0),
            "sleep_hrs": (7.4, 0.7, 6.0, 9.0),
        },
        CATEGORY_AVERAGE: {
            "attendance": (74.0, 6.0, 60.0, 88.0),
            "prev_marks": (66.0, 6.5, 52.0, 78.0),
            "assignment": (68.0, 6.0, 54.0, 82.0),
            "internals": (65.0, 6.0, 50.0, 78.0),
            "study_hrs": (3.4, 0.8, 1.8, 5.2),
            "sleep_hrs": (6.9, 0.8, 5.2, 8.5),
        },
        CATEGORY_AT_RISK: {
            "attendance": (55.0, 7.5, 35.0, 68.0),
            "prev_marks": (44.0, 6.5, 25.0, 58.0),
            "assignment": (46.0, 7.0, 28.0, 60.0),
            "internals": (42.0, 6.5, 25.0, 55.0),
            "study_hrs": (1.6, 0.6, 0.5, 3.0),
            "sleep_hrs": (5.4, 1.0, 3.5, 7.5),
        },
    }

    # Generate labelled historical rows
    for category in [CATEGORY_GOOD, CATEGORY_AVERAGE, CATEGORY_AT_RISK]:
        params = profiles[category]
        for _ in range(num_labelled_per_class):
            att = np.clip(rng.normal(*params["attendance"][:2]), *params["attendance"][2:])
            prev = np.clip(rng.normal(*params["prev_marks"][:2]), *params["prev_marks"][2:])
            assign = np.clip(rng.normal(*params["assignment"][:2]), *params["assignment"][2:])
            internal = np.clip(rng.normal(*params["internals"][:2]), *params["internals"][2:])
            study = np.clip(rng.normal(*params["study_hrs"][:2]), *params["study_hrs"][2:])
            sleep = np.clip(rng.normal(*params["sleep_hrs"][:2]), *params["sleep_hrs"][2:])

            records.append({
                COL_STUDENT_ID: f"STU-{id_counter}",
                COL_ATTENDANCE: round(float(att), 1),
                COL_PREVIOUS_MARKS: round(float(prev), 1),
                COL_ASSIGNMENT_AVG: round(float(assign), 1),
                COL_INTERNAL_MARKS: round(float(internal), 1),
                COL_STUDY_HOURS: round(float(study), 1),
                COL_SLEEP_HOURS: round(float(sleep), 1),
                COL_PERFORMANCE_CATEGORY: category,
            })
            id_counter += 1

    # Generate unlabelled rows (students awaiting performance prediction)
    # Mix of borderline and typical profiles
    unlabelled_categories = rng.choice(
        [CATEGORY_GOOD, CATEGORY_AVERAGE, CATEGORY_AT_RISK], size=num_unlabelled
    )
    for category in unlabelled_categories:
        params = profiles[category]
        att = np.clip(rng.normal(*params["attendance"][:2]), *params["attendance"][2:])
        prev = np.clip(rng.normal(*params["prev_marks"][:2]), *params["prev_marks"][2:])
        assign = np.clip(rng.normal(*params["assignment"][:2]), *params["assignment"][2:])
        internal = np.clip(rng.normal(*params["internals"][:2]), *params["internals"][2:])
        study = np.clip(rng.normal(*params["study_hrs"][:2]), *params["study_hrs"][2:])
        sleep = np.clip(rng.normal(*params["sleep_hrs"][:2]), *params["sleep_hrs"][2:])

        records.append({
            COL_STUDENT_ID: f"STU-{id_counter}",
            COL_ATTENDANCE: round(float(att), 1),
            COL_PREVIOUS_MARKS: round(float(prev), 1),
            COL_ASSIGNMENT_AVG: round(float(assign), 1),
            COL_INTERNAL_MARKS: round(float(internal), 1),
            COL_STUDY_HOURS: round(float(study), 1),
            COL_SLEEP_HOURS: round(float(sleep), 1),
            COL_PERFORMANCE_CATEGORY: None,  # Blank for prediction
        })
        id_counter += 1

    df = pd.DataFrame(records)
    # Shuffle to mix labelled and unlabelled naturally
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def save_synthetic_excel(
    file_path: str = "data/sample_students.xlsx",
    num_labelled_per_class: int = 30,
    num_unlabelled: int = 25,
) -> str:
    """Generates synthetic student data and saves as an Excel workbook."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = generate_synthetic_students(
        num_labelled_per_class=num_labelled_per_class,
        num_unlabelled=num_unlabelled,
    )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Student_Academic_Data"

    # Styling
    header_fill = PatternFill(start_color="0F5132", end_color="0F5132", fill_type="solid")
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin", color="E2E8F0"),
        right=Side(style="thin", color="E2E8F0"),
        top=Side(style="thin", color="E2E8F0"),
        bottom=Side(style="thin", color="E2E8F0"),
    )
    center_align = Alignment(horizontal="center", vertical="center")

    cols = list(df.columns)
    ws.append(cols)

    for col_idx in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    # Add rows
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        row_vals = list(row)
        # Convert None to empty string
        row_vals = ["" if val is None or (isinstance(val, float) and np.isnan(val)) else val for val in row_vals]
        ws.append(row_vals)
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = thin_border
            cell.alignment = center_align

    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 16)

    wb.save(file_path)
    return file_path


if __name__ == "__main__":
    path = save_synthetic_excel()
    print(f"Sample synthetic dataset generated at: {path}")
