# actions.py
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, Union


class InspectDataAction(BaseModel):
    action_type: Literal["inspect_data"]
    target: Literal["head", "dtypes", "describe", "value_counts", "null_check", "shape"]
    column: Optional[str] = None


class RunCodeAction(BaseModel):
    action_type: Literal["run_code"]
    code: str


class ApplyFixAction(BaseModel):
    action_type: Literal["apply_fix"]
    fix_type: Literal[
        "drop_leaky_column",
        "resample_class_balance",
        "set_hyperparameter",
        "fix_scaler_placement",
        "fix_train_test_split",
        "fix_missing_value_handling",
    ]
    parameters: Dict[str, Any] = {}


class EvaluateModelAction(BaseModel):
    action_type: Literal["evaluate_model"]


class SubmitDiagnosisAction(BaseModel):
    action_type: Literal["submit_diagnosis"]
    bug_type: Literal[
        "data_leakage",
        "class_imbalance",
        "wrong_hyperparameter",
        "scaling_error",
        "train_test_contamination",
        "missing_value_handling",
    ]
    explanation: str


Action = Union[
    InspectDataAction,
    RunCodeAction,
    ApplyFixAction,
    EvaluateModelAction,
    SubmitDiagnosisAction,
]
