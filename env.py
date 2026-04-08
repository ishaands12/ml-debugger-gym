# env.py
# Main environment class for the Meta PyTorch OpenEnv Hackathon.
# Implements the openenv_core.Environment API.

import io
import sys
import time
import traceback
from typing import Optional

from actions import (
    Action,
    ApplyFixAction,
    EvaluateModelAction,
    InspectDataAction,
    RunCodeAction,
    SubmitDiagnosisAction,
)
from dataset_generator import generate_broken_pipeline
from grader import MLDebuggerRubric
from observations import CodeExecutionResult, ModelMetrics, Observation

try:
    from openenv_core import Environment as BaseEnvironment
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="openenv_core")
except ImportError:
    class BaseEnvironment:
        rubric = None
        def __init__(self, *args, **kwargs):
            pass


class MLDebuggerEnv(BaseEnvironment):
    """
    ML Experiment Debugger Gym

    Agent acts as a data scientist debugging a broken ML pipeline.
    Must inspect data, identify the injected bug, apply a fix, verify
    the accuracy improvement, and submit a formal diagnosis.

    Compatible with openenv_core.Environment API.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    AVAILABLE_ACTIONS = [
        "inspect_data",
        "run_code",
        "apply_fix",
        "evaluate_model",
        "submit_diagnosis",
    ]

    def __init__(self):
        self._rubric = MLDebuggerRubric()
        super().__init__(rubric=None)  # We handle rewards ourselves
        self.max_steps = 15
        self._obs: Optional[Observation] = None
        self._ground_truth_bug: Optional[str] = None
        self._wrong_submissions: int = 0
        self._dataset = None
        self._pipeline = None

    # ------------------------------------------------------------------
    # OpenEnv public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              difficulty: int = 1, **kwargs) -> Observation:
        """
        Start a new episode with a fresh broken pipeline.

        Args:
            seed: Optional random seed for reproducibility
            episode_id: Optional episode ID (auto-generated if None)
            difficulty: 1=Easy, 2=Medium, 3=Hard, 4=Expert

        Returns:
            Initial Observation.
        """
        difficulty = max(1, min(4, int(difficulty)))
        pipeline_data = generate_broken_pipeline(difficulty=difficulty, seed=seed)

        self._dataset = pipeline_data["dataset"]
        self._pipeline = pipeline_data["pipeline"]
        self._ground_truth_bug = pipeline_data["bug_type"]
        self._wrong_submissions = 0

        baseline = self._run_evaluation()
        target_accuracy = min(baseline.accuracy + 0.10 + (difficulty * 0.05), 0.95)

        self._obs = Observation(
            done=False,
            reward=None,
            step=0,
            max_steps=self.max_steps,
            task_description=pipeline_data["task_description"],
            current_metrics=baseline,
            baseline_metrics=baseline,
            target_accuracy=round(target_accuracy, 3),
            last_action_result=None,
            available_actions=self.AVAILABLE_ACTIONS,
            hints_used=0,
            episode_id=episode_id or pipeline_data["episode_id"],
        )
        return self._obs

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        """
        Advance the environment by one step.

        Args:
            action: Parsed Action object (one of the 5 action types)
            timeout_s: Ignored (for API compatibility)

        Returns:
            Updated Observation with reward and done embedded.
        """
        assert self._obs is not None, "Call reset() before step()"

        self._obs.step += 1
        result = self._execute_action(action)
        self._obs.last_action_result = result

        reward = self._rubric.forward(action, self)
        done, info = self._check_termination(action)

        if done and info.get("reason") == "timeout":
            reward -= 0.5

        self._obs.done = done
        self._obs.reward = round(reward, 4)
        self._obs.metadata = info

        return self._obs

    def state(self) -> Optional[Observation]:
        """Return current observation (read-only)."""
        return self._obs

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _execute_action(self, action: Action) -> Optional[CodeExecutionResult]:
        if action.action_type == "inspect_data":
            return self._handle_inspect(action)
        elif action.action_type == "run_code":
            return self._handle_run_code(action)
        elif action.action_type == "apply_fix":
            return self._handle_apply_fix(action)
        elif action.action_type == "evaluate_model":
            self._obs.current_metrics = self._run_evaluation()
            return None
        elif action.action_type == "submit_diagnosis":
            if action.bug_type != self._ground_truth_bug:
                self._wrong_submissions += 1
            return None
        return None

    # ------------------------------------------------------------------
    # ML evaluation
    # ------------------------------------------------------------------

    def _run_evaluation(self) -> ModelMetrics:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        import numpy as np

        X = self._pipeline["X"]
        y = self._pipeline["y"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        params = {k: v for k, v in self._pipeline.get("hyperparams", {}).items()
                  if k in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf")}
        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        val_f1 = f1_score(y_val, clf.predict(X_val), average="weighted", zero_division=0)
        classes, counts = np.unique(y, return_counts=True)

        return ModelMetrics(
            accuracy=round(val_acc, 4),
            f1_score=round(val_f1, 4),
            train_accuracy=round(train_acc, 4),
            val_accuracy=round(val_acc, 4),
            class_distribution={str(int(k)): int(v) for k, v in zip(classes, counts)},
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(self, action: Action) -> tuple:
        if (
            action.action_type == "submit_diagnosis"
            and action.bug_type == self._ground_truth_bug
            and self._obs.current_metrics.accuracy >= self._obs.target_accuracy
        ):
            return True, {"reason": "success", "bug_found": True}

        if self._wrong_submissions >= 3:
            return True, {"reason": "too_many_wrong_submissions"}

        if self._obs.step >= self.max_steps:
            return True, {"reason": "timeout"}

        return False, {}

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_inspect(self, action: InspectDataAction) -> CodeExecutionResult:
        buf = io.StringIO()
        df = self._dataset
        try:
            if action.target == "head":
                buf.write(df.head(10).to_string())
            elif action.target == "dtypes":
                buf.write(str(df.dtypes))
            elif action.target == "describe":
                buf.write(df.describe().to_string())
            elif action.target == "value_counts":
                col = action.column or "target"
                buf.write(df[col].value_counts().to_string())
            elif action.target == "null_check":
                buf.write(df.isnull().sum().to_string())
            elif action.target == "shape":
                buf.write(str(df.shape))
            return CodeExecutionResult(stdout=buf.getvalue(), stderr="", execution_time_ms=10)
        except Exception as e:
            return CodeExecutionResult(stdout="", stderr=str(e), execution_time_ms=0)

    def _handle_run_code(self, action: RunCodeAction) -> CodeExecutionResult:
        stdout_buf = io.StringIO()
        start = time.time()
        local_vars = {
            "df": self._dataset.copy(),
            "pipeline": self._pipeline,
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
        }
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_buf
            exec(action.code, {"__builtins__": __builtins__}, local_vars)  # noqa: S102
            sys.stdout = old_stdout
            elapsed = int((time.time() - start) * 1000)
            return CodeExecutionResult(
                stdout=stdout_buf.getvalue(), stderr="", execution_time_ms=elapsed
            )
        except Exception:
            sys.stdout = old_stdout
            return CodeExecutionResult(
                stdout="", stderr=traceback.format_exc(), execution_time_ms=0
            )

    def _handle_apply_fix(self, action: ApplyFixAction) -> CodeExecutionResult:
        params = action.parameters
        try:
            if action.fix_type == "drop_leaky_column":
                col = params.get("column")
                if col and col in self._dataset.columns:
                    self._dataset = self._dataset.drop(columns=[col])
                    self._pipeline["X"] = self._dataset.drop(columns=["target"])
                    return CodeExecutionResult(
                        stdout=f"Dropped leaky column: {col}", stderr="", execution_time_ms=5
                    )
                return CodeExecutionResult(
                    stdout="", stderr=f"Column '{col}' not found.", execution_time_ms=0
                )

            elif action.fix_type == "resample_class_balance":
                strategy = params.get("strategy", "oversample")
                X = self._pipeline["X"]
                y = self._pipeline["y"]
                import numpy as np

                if strategy in ("oversample", "smote"):
                    try:
                        from imblearn.over_sampling import SMOTE, RandomOverSampler
                        sampler = SMOTE(random_state=42) if strategy == "smote" else RandomOverSampler(random_state=42)
                        X_res, y_res = sampler.fit_resample(X, y)
                    except ImportError:
                        classes, counts = np.unique(y, return_counts=True)
                        max_count = counts.max()
                        import pandas as pd
                        parts_X, parts_y = [], []
                        for c in classes:
                            mask = y == c
                            idx = np.random.choice(mask.sum(), max_count, replace=True)
                            parts_X.append(X[mask].iloc[idx])
                            parts_y.append(np.full(max_count, c))
                        X_res = pd.concat(parts_X).reset_index(drop=True)
                        y_res = np.concatenate(parts_y)
                else:
                    from sklearn.utils import resample
                    classes = np.unique(y)
                    min_count = min(int(np.sum(y == c)) for c in classes)
                    import pandas as pd
                    parts_X, parts_y = [], []
                    for c in classes:
                        mask = y == c
                        parts_X.append(resample(X[mask], n_samples=min_count, random_state=42))
                        parts_y.append(np.full(min_count, c))
                    X_res = pd.concat(parts_X).reset_index(drop=True)
                    y_res = np.concatenate(parts_y)

                self._pipeline["X"] = X_res
                self._pipeline["y"] = y_res
                return CodeExecutionResult(
                    stdout=f"Resampled (strategy={strategy}). New shape: {X_res.shape}",
                    stderr="", execution_time_ms=50,
                )

            elif action.fix_type == "set_hyperparameter":
                key = params.get("key") or params.get("param_name")
                val = params.get("value")
                if key:
                    self._pipeline.setdefault("hyperparams", {})[key] = val
                    return CodeExecutionResult(
                        stdout=f"Set {key} = {val}", stderr="", execution_time_ms=5
                    )
                return CodeExecutionResult(stdout="", stderr="Missing 'key'.", execution_time_ms=0)

            elif action.fix_type == "fix_scaler_placement":
                self._pipeline["scaler_fitted_on_full"] = False
                self._pipeline["X"] = self._dataset.drop(columns=["target"]).copy()
                return CodeExecutionResult(
                    stdout="Scaler placement fixed (will fit on train only).",
                    stderr="", execution_time_ms=10,
                )

            elif action.fix_type == "fix_train_test_split":
                test_size = params.get("test_size", 0.2)
                self._pipeline["test_size"] = test_size
                return CodeExecutionResult(
                    stdout=f"Split fixed (test_size={test_size}).", stderr="", execution_time_ms=5
                )

            elif action.fix_type == "fix_missing_value_handling":
                self._pipeline["bad_imputer"] = False
                return CodeExecutionResult(
                    stdout="Imputer will fit on train only.", stderr="", execution_time_ms=5
                )

        except Exception:
            return CodeExecutionResult(
                stdout="", stderr=traceback.format_exc(), execution_time_ms=0
            )

        return CodeExecutionResult(stdout="Fix applied.", stderr="", execution_time_ms=5)
