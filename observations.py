# observations.py
from typing import Optional, List, Dict
from pydantic import BaseModel

try:
    from openenv.core import Observation as _BaseObs
    _BASE = _BaseObs
except ImportError:
    try:
        from openenv_core import Observation as _BaseObs
        _BASE = _BaseObs
    except ImportError:
        _BASE = BaseModel


class ModelMetrics(BaseModel):
    accuracy: float
    f1_score: float
    train_accuracy: float
    val_accuracy: float
    class_distribution: Dict[str, int]
    # PyTorch training diagnostics — empty for sklearn models
    loss_curve: List[float] = []           # train loss per epoch
    gradient_norm: Optional[float] = None  # L2 norm of gradients at final step


class CodeExecutionResult(BaseModel):
    stdout: str
    stderr: str
    execution_time_ms: int


class Observation(_BASE):
    """
    Environment observation. Extends openenv_core.Observation with
    ML Debugger Gym fields (inherits done, reward, metadata from parent).
    """
    step: int = 0
    max_steps: int = 15
    task_description: str = ""
    current_metrics: Optional[ModelMetrics] = None
    baseline_metrics: Optional[ModelMetrics] = None
    target_accuracy: float = 0.0
    last_action_result: Optional[CodeExecutionResult] = None
    available_actions: List[str] = []
    hints_used: int = 0
    episode_id: str = ""
