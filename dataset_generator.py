# dataset_generator.py
# Procedurally generates broken ML pipelines for each episode.
# Each call returns a fresh pipeline with one injected bug.

import uuid
import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def generate_broken_pipeline(difficulty: int = 1, seed: int = None) -> dict:
    """
    Generate a broken ML pipeline for an episode.

    Args:
        difficulty: 1=Easy, 2=Medium, 3=Hard, 4=Expert
        seed: Optional random seed for reproducibility

    Returns:
        {
            "dataset": pd.DataFrame,        # full dataset including target column
            "pipeline": dict,               # {"X": DataFrame, "y": array, "hyperparams": dict}
            "bug_type": str,                # ground truth bug label
            "task_description": str,        # natural language description shown to agent
            "episode_id": str,              # UUID for logging
        }
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = np.random.RandomState(seed)
    episode_id = str(uuid.uuid4())

    bug_map = {
        1: "data_leakage",
        2: "class_imbalance",
        3: "wrong_hyperparameter",
        4: "scaling_error",
    }

    # For difficulty 3+ inject a second bug as well (Hard/Expert)
    primary_bug = bug_map.get(difficulty, "data_leakage")

    # ---- Base dataset ----
    n_samples = 1000 if difficulty < 4 else 800
    n_features = 10 if difficulty < 3 else 15
    n_informative = 5 if difficulty < 3 else 6
    n_redundant = 2

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    # Expert: obfuscate column names
    if difficulty == 4:
        feature_names = [f"x{i:02d}" for i in range(n_features)]

    df = pd.DataFrame(X_raw, columns=feature_names)
    df["target"] = y

    # ---- Inject primary bug ----
    pipeline = {
        "X": df.drop(columns=["target"]).copy(),
        "y": y.copy(),
        "hyperparams": {"n_estimators": 100, "max_depth": None},
        "bug_type": primary_bug,
        "scaler_fitted_on_full": False,
        "contaminated_split": False,
        "bad_imputer": False,
        "leaky_column": None,
    }

    if primary_bug == "data_leakage":
        pipeline = _inject_data_leakage(df, pipeline, rng, difficulty)
    elif primary_bug == "class_imbalance":
        df, pipeline = _inject_class_imbalance(df, pipeline, rng)
    elif primary_bug == "wrong_hyperparameter":
        pipeline = _inject_wrong_hyperparameter(pipeline, rng)
    elif primary_bug == "scaling_error":
        pipeline = _inject_scaling_error(df, pipeline)

    # Hard/Expert: add a second bug
    if difficulty >= 3:
        secondary_bugs = [b for b in ["class_imbalance", "wrong_hyperparameter"] if b != primary_bug]
        second = rng.choice(secondary_bugs)
        if second == "class_imbalance":
            df, pipeline = _inject_class_imbalance(df, pipeline, rng, ratio=0.92)
        elif second == "wrong_hyperparameter":
            pipeline = _inject_wrong_hyperparameter(pipeline, rng)

    task_description = _build_task_description(primary_bug, difficulty, pipeline)

    return {
        "dataset": df,
        "pipeline": pipeline,
        "bug_type": primary_bug,
        "task_description": task_description,
        "episode_id": episode_id,
    }


# ---------------------------------------------------------------------------
# Bug injectors
# ---------------------------------------------------------------------------

def _inject_data_leakage(df, pipeline, rng, difficulty):
    """Add a column perfectly correlated with target (correlation > 0.99)."""
    noise = rng.normal(0, 0.01, size=len(df))
    leaky_col = "target_encoded" if difficulty < 4 else "enc_y"
    df[leaky_col] = df["target"].astype(float) + noise
    pipeline["X"] = df.drop(columns=["target"]).copy()
    pipeline["leaky_column"] = leaky_col
    return pipeline


def _inject_class_imbalance(df, pipeline, rng, ratio=0.95):
    """Create severe class imbalance (ratio : 1-ratio)."""
    majority = df[df["target"] == 0]
    minority = df[df["target"] == 1]

    n_majority = int(len(df) * ratio)
    n_minority = len(df) - n_majority

    majority_sample = majority.sample(n=min(n_majority, len(majority)), random_state=42, replace=len(majority) < n_majority)
    minority_sample = minority.sample(n=min(n_minority, len(minority)), random_state=42, replace=len(minority) < n_minority)

    df_imbalanced = pd.concat([majority_sample, minority_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    pipeline["X"] = df_imbalanced.drop(columns=["target"]).copy()
    pipeline["y"] = df_imbalanced["target"].values
    return df_imbalanced, pipeline


def _inject_wrong_hyperparameter(pipeline, rng):
    """Set a pathological hyperparameter value."""
    bad_params = [
        {"max_depth": 1},
        {"n_estimators": 1},
        {"min_samples_split": 500},
    ]
    chosen = bad_params[rng.randint(0, len(bad_params))]
    pipeline["hyperparams"].update(chosen)
    return pipeline


def _inject_scaling_error(df, pipeline):
    """Fit StandardScaler on full dataset before train/test split."""
    scaler = StandardScaler()
    X = df.drop(columns=["target"])
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    pipeline["X"] = X_scaled
    pipeline["scaler_fitted_on_full"] = True
    return pipeline


# ---------------------------------------------------------------------------
# Task descriptions
# ---------------------------------------------------------------------------

_HINTS = {
    "data_leakage": {
        1: "The pipeline is producing suspiciously high validation accuracy. A column in the feature matrix may be directly correlated with the target label. Inspect the features carefully.",
        2: "Model performance looks too good to be true. Investigate feature correlations — something in the data may be giving away the answer.",
        3: "Two bugs detected. The pipeline is underperforming. Investigate systematically.",
        4: "Pipeline is underperforming. No additional hints available.",
    },
    "class_imbalance": {
        1: "The model achieves high accuracy but performs poorly on one class. Inspect the class distribution — the dataset may be severely imbalanced.",
        2: "Accuracy is misleadingly high. Look at the F1 score and class distribution.",
        3: "Two bugs detected. The pipeline is underperforming. Investigate systematically.",
        4: "Pipeline is underperforming. No additional hints available.",
    },
    "wrong_hyperparameter": {
        1: "The model is severely underfitting. A hyperparameter may be set to a pathological value that prevents the model from learning.",
        2: "Training accuracy is very low. Investigate the model hyperparameters.",
        3: "Two bugs detected. The pipeline is underperforming. Investigate systematically.",
        4: "Pipeline is underperforming. No additional hints available.",
    },
    "scaling_error": {
        1: "The validation metrics look slightly inflated. The preprocessing pipeline may have a data leakage issue related to feature scaling.",
        2: "Investigate whether the scaler was fitted correctly — fit only on training data.",
        3: "Two bugs detected. The pipeline is underperforming. Investigate systematically.",
        4: "Pipeline is underperforming. No additional hints available.",
    },
}

def _build_task_description(bug_type, difficulty, pipeline):
    return _HINTS.get(bug_type, {}).get(difficulty, "The ML pipeline is underperforming. Diagnose the bug and fix it.")
