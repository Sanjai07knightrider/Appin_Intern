"""Constants and configuration settings for EduPredict AI."""

# Dataset Columns
COL_STUDENT_ID = "student_id"
COL_ATTENDANCE = "attendance_pct"
COL_PREVIOUS_MARKS = "previous_marks_pct"
COL_ASSIGNMENT_AVG = "assignment_avg_pct"
COL_INTERNAL_MARKS = "internal_marks_pct"
COL_STUDY_HOURS = "study_hours_per_day"
COL_SLEEP_HOURS = "sleep_hours_per_day"
COL_PERFORMANCE_CATEGORY = "performance_category"

# Added output columns
COL_PREDICTED_CATEGORY = "predicted_category"
COL_CONFIDENCE = "confidence_score"
COL_IS_PREDICTION = "is_predicted"

REQUIRED_COLUMNS = [
    COL_STUDENT_ID,
    COL_ATTENDANCE,
    COL_PREVIOUS_MARKS,
    COL_ASSIGNMENT_AVG,
    COL_INTERNAL_MARKS,
    COL_STUDY_HOURS,
    COL_SLEEP_HOURS,
    COL_PERFORMANCE_CATEGORY,
]

FEATURE_COLUMNS = [
    COL_ATTENDANCE,
    COL_PREVIOUS_MARKS,
    COL_ASSIGNMENT_AVG,
    COL_INTERNAL_MARKS,
    COL_STUDY_HOURS,
    COL_SLEEP_HOURS,
]

FEATURE_PRETTY_NAMES = {
    COL_ATTENDANCE: "Attendance (%)",
    COL_PREVIOUS_MARKS: "Previous Marks (%)",
    COL_ASSIGNMENT_AVG: "Assignment Average (%)",
    COL_INTERNAL_MARKS: "Internal Marks (%)",
    COL_STUDY_HOURS: "Daily Study Hours",
    COL_SLEEP_HOURS: "Daily Sleep Hours",
}

# Numeric Validation Bounds
NUMERIC_RANGES = {
    COL_ATTENDANCE: (0.0, 100.0),
    COL_PREVIOUS_MARKS: (0.0, 100.0),
    COL_ASSIGNMENT_AVG: (0.0, 100.0),
    COL_INTERNAL_MARKS: (0.0, 100.0),
    COL_STUDY_HOURS: (0.0, 24.0),
    COL_SLEEP_HOURS: (0.0, 24.0),
}

# Valid Category Labels
CATEGORY_AT_RISK = "At Risk"
CATEGORY_AVERAGE = "Average"
CATEGORY_GOOD = "Good"

VALID_CATEGORIES = [CATEGORY_AT_RISK, CATEGORY_AVERAGE, CATEGORY_GOOD]

# Category Normalization Map (case-insensitive & whitespace trimmed)
CATEGORY_NORMALIZATION_MAP = {
    "at risk": CATEGORY_AT_RISK,
    "at-risk": CATEGORY_AT_RISK,
    "atrisk": CATEGORY_AT_RISK,
    "risk": CATEGORY_AT_RISK,
    "average": CATEGORY_AVERAGE,
    "avg": CATEGORY_AVERAGE,
    "good": CATEGORY_GOOD,
}

# Color Palette (Emerald / Mint / Forest Green Theme)
COLOR_PRIMARY = "#0F5132"       # Deep forest green
COLOR_ACCENT = "#10B981"        # Vivid emerald mint
COLOR_ACCENT_HOVER = "#059669"  # Darker emerald
COLOR_BG_LIGHT = "#F8FAFC"      # Clean slate white
COLOR_CARD_BG = "#FFFFFF"       # Pure white
COLOR_BORDER = "#E2E8F0"        # Subtle slate border

# Category Colors
CATEGORY_COLORS = {
    CATEGORY_AT_RISK: "#EF4444",   # Clean warning red / coral
    CATEGORY_AVERAGE: "#F59E0B",   # Warm amber
    CATEGORY_GOOD: "#10B981",      # Emerald green
}

CATEGORY_BG_COLORS = {
    CATEGORY_AT_RISK: "rgba(239, 68, 68, 0.12)",
    CATEGORY_AVERAGE: "rgba(245, 158, 11, 0.12)",
    CATEGORY_GOOD: "rgba(16, 185, 129, 0.12)",
}
