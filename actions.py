# actions.py
from typing import Annotated, Any, Dict, Literal, Optional, Union
from pydantic import Field

try:
    from openenv.core import Action as _BaseAction
    _BASE = _BaseAction
except ImportError:
    try:
        from openenv_core import Action as _BaseAction
        _BASE = _BaseAction
    except ImportError:
        from pydantic import BaseModel
        _BASE = BaseModel


class InspectDataAction(_BASE):
    action_type: Literal["inspect_data"] = "inspect_data"
    target: Literal["head", "dtypes", "describe", "value_counts", "null_check", "shape"]
    column: Optional[str] = None


class RunCodeAction(_BASE):
    action_type: Literal["run_code"] = "run_code"
    code: str


class ApplyFixAction(_BASE):
    action_type: Literal["apply_fix"] = "apply_fix"
    fix_type: Literal[
        "drop_leaky_column",
        "resample_class_balance",
        "set_hyperparameter",
        "fix_scaler_placement",
        "fix_train_test_split",
        "fix_missing_value_handling",
        "fix_learning_rate",
        "fix_weight_initialization",
        "fix_batch_normalization",
    ]
    parameters: Dict[str, Any] = {}


class EvaluateModelAction(_BASE):
    action_type: Literal["evaluate_model"] = "evaluate_model"


class SubmitDiagnosisAction(_BASE):
    action_type: Literal["submit_diagnosis"] = "submit_diagnosis"
    bug_type: Literal[
        "data_leakage",
        "class_imbalance",
        "wrong_hyperparameter",
        "scaling_error",
        "train_test_contamination",
        "missing_value_handling",
        "wrong_learning_rate",
        "exploding_gradients",
        "missing_batch_normalization",
    ]
    explanation: str


# Discriminated union — the single action type exposed to OpenEnv
Action = Annotated[
    Union[
        InspectDataAction,
        RunCodeAction,
        ApplyFixAction,
        EvaluateModelAction,
        SubmitDiagnosisAction,
    ],
    Field(discriminator="action_type"),
]
