"""Machine Learning Pipeline for EduPredict AI.

Implements preprocessing, holdout evaluation, model comparison, final retraining,
and prediction with confidence estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import (
    COL_CONFIDENCE,
    COL_IS_PREDICTION,
    COL_PERFORMANCE_CATEGORY,
    COL_PREDICTED_CATEGORY,
    COL_STUDENT_ID,
    FEATURE_COLUMNS,
    FEATURE_PRETTY_NAMES,
    VALID_CATEGORIES,
)


@dataclass
class ModelMetrics:
    """Holdout evaluation metrics for a single model."""
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    classification_report_dict: Dict[str, Any]
    confusion_matrix: List[List[int]]
    classes: List[str]
    train_size: int
    test_size: int


@dataclass
class PipelineTrainingResult:
    """Complete results from ML pipeline training and holdout evaluation."""
    best_model_name: str
    best_model_pipeline: Pipeline
    models_metrics: Dict[str, ModelMetrics]
    feature_importances: Dict[str, float]
    labelled_count: int
    test_size: int
    classes: List[str]


def evaluate_model_pipeline(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    classes: List[str],
) -> Tuple[ModelMetrics, Pipeline]:
    """Trains pipeline on training fold and evaluates on holdout test set."""
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    # Compute holdout metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    metrics = ModelMetrics(
        model_name=name,
        accuracy=acc,
        precision_macro=prec_macro,
        recall_macro=rec_macro,
        f1_macro=f1_macro,
        classification_report_dict=report,
        confusion_matrix=cm,
        classes=classes,
        train_size=len(x_train),
        test_size=len(x_test),
    )
    return metrics, pipeline


def train_and_evaluate_pipeline(
    labelled_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PipelineTrainingResult:
    """Trains baseline (Logistic Regression) and enhanced (Random Forest) models.
    
    1. Validates minimum training requirements.
    2. Performs stratified train/test split.
    3. Evaluates both models on held-out test data.
    4. Identifies the superior model by macro F1-score.
    5. Retrains the selected best model on all labelled data.
    6. Extracts feature importance / weights.
    """
    if len(labelled_df) < 6:
        raise ValueError(
            f"Insufficient labelled historical data: found {len(labelled_df)} records. "
            "A minimum of 6 labelled records (ideally 15+) across categories is required "
            "to perform a meaningful train/test split and holdout evaluation."
        )

    x_all = labelled_df[FEATURE_COLUMNS].copy()
    y_all = labelled_df[COL_PERFORMANCE_CATEGORY].copy()

    unique_classes = sorted(list(y_all.unique()))
    if len(unique_classes) < 2:
        raise ValueError(
            f"The dataset only contains one category ({unique_classes[0]}). "
            "Classification models require at least two distinct categories in historical training data."
        )

    # Determine if stratified split is possible (each class needs >= 2 samples)
    class_counts = y_all.value_counts()
    min_count = class_counts.min()
    can_stratify = min_count >= 2

    # Split into train and holdout test set
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all if can_stratify else None,
    )

    # Define Candidate 1: Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])

    # Define Candidate 2: Random Forest Pipeline
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
        )),
    ])

    # Evaluate Candidate 1
    lr_metrics, _ = evaluate_model_pipeline(
        name="Logistic Regression (Baseline)",
        pipeline=lr_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        classes=unique_classes,
    )

    # Evaluate Candidate 2
    rf_metrics, _ = evaluate_model_pipeline(
        name="Random Forest Classifier",
        pipeline=rf_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        classes=unique_classes,
    )

    models_metrics = {
        lr_metrics.model_name: lr_metrics,
        rf_metrics.model_name: rf_metrics,
    }

    # Select best model based on macro F1, tie-breaker on accuracy
    if rf_metrics.f1_macro >= lr_metrics.f1_macro:
        best_name = rf_metrics.model_name
        best_pipeline_template = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
            )),
        ])
    else:
        best_name = lr_metrics.model_name
        best_pipeline_template = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])

    # Retrain final model on ALL valid labelled historical rows
    best_pipeline_template.fit(x_all, y_all)

    # Extract feature importances
    feature_importances: Dict[str, float] = {}
    clf = best_pipeline_template.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        raw_importances = clf.feature_importances_
        for col, imp in zip(FEATURE_COLUMNS, raw_importances):
            feature_importances[FEATURE_PRETTY_NAMES.get(col, col)] = float(imp)
    elif hasattr(clf, "coef_"):
        # For multiclass logistic regression, average absolute coefficients across classes
        raw_importances = np.mean(np.abs(clf.coef_), axis=0)
        total = np.sum(raw_importances)
        normalized = raw_importances / total if total > 0 else raw_importances
        for col, imp in zip(FEATURE_COLUMNS, normalized):
            feature_importances[FEATURE_PRETTY_NAMES.get(col, col)] = float(imp)

    # Sort descending
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    )

    return PipelineTrainingResult(
        best_model_name=best_name,
        best_model_pipeline=best_pipeline_template,
        models_metrics=models_metrics,
        feature_importances=feature_importances,
        labelled_count=len(labelled_df),
        test_size=len(x_test),
        classes=unique_classes,
    )


def generate_predictions(
    model_pipeline: Pipeline,
    df_to_predict: pd.DataFrame,
) -> pd.DataFrame:
    """Generates predicted performance category and confidence probability for given rows.
    
    Returns a copy of df_to_predict with COL_PREDICTED_CATEGORY and COL_CONFIDENCE attached.
    """
    if df_to_predict.empty:
        result = df_to_predict.copy()
        result[COL_PREDICTED_CATEGORY] = []
        result[COL_CONFIDENCE] = []
        return result

    result = df_to_predict.copy()
    x = result[FEATURE_COLUMNS].copy()

    # Predict category
    preds = model_pipeline.predict(x)
    result[COL_PREDICTED_CATEGORY] = preds

    # Predict probability confidence
    if hasattr(model_pipeline, "predict_proba"):
        probs = model_pipeline.predict_proba(x)
        max_probs = np.max(probs, axis=1)
        result[COL_CONFIDENCE] = np.round(max_probs, 4)
    else:
        result[COL_CONFIDENCE] = 1.0

    return result


def apply_predictions_to_full_dataset(
    model_pipeline: Pipeline,
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Produces the final master dataset:
    - Retains original student data.
    - Preserves historical labels in performance_category.
    - For unlabelled rows: fills predicted_category and confidence_score.
    - For historical rows: predicted_category mirrors performance_category, confidence is 1.0 (or model self-score).
    """
    master_df = cleaned_df.copy()

    unlabelled_mask = master_df[COL_IS_PREDICTION] == True
    unlabelled_rows = master_df[unlabelled_mask]

    # Initialize prediction columns
    master_df[COL_PREDICTED_CATEGORY] = master_df[COL_PERFORMANCE_CATEGORY]
    master_df[COL_CONFIDENCE] = np.nan

    if not unlabelled_rows.empty:
        predicted_subset = generate_predictions(model_pipeline, unlabelled_rows)
        master_df.loc[unlabelled_mask, COL_PREDICTED_CATEGORY] = predicted_subset[
            COL_PREDICTED_CATEGORY
        ].values
        master_df.loc[unlabelled_mask, COL_CONFIDENCE] = predicted_subset[
            COL_CONFIDENCE
        ].values

    # For labelled rows, we can also compute the model's posterior confidence if desired
    labelled_mask = ~unlabelled_mask
    if labelled_mask.any() and hasattr(model_pipeline, "predict_proba"):
        labelled_x = master_df.loc[labelled_mask, FEATURE_COLUMNS]
        labelled_probs = model_pipeline.predict_proba(labelled_x)
        master_df.loc[labelled_mask, COL_CONFIDENCE] = np.round(
            np.max(labelled_probs, axis=1), 4
        )

    return master_df
