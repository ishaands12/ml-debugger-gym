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
    from openenv.core import Environment as BaseEnvironment
except ImportError:
    try:
        from openenv_core import Environment as BaseEnvironment
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
        super().__init__(rubric=self._rubric)
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
        difficulty = max(1, min(7, int(difficulty)))
        pipeline_data = generate_broken_pipeline(difficulty=difficulty, seed=seed)

        self._dataset = pipeline_data["dataset"]
        self._pipeline = pipeline_data["pipeline"]
        self._ground_truth_bug = pipeline_data["bug_type"]
        self._wrong_submissions = 0
        self._rubric.set_ground_truth(self._ground_truth_bug)
        self._reset_rubric()

        baseline = self._run_evaluation()
        # For data_leakage: removing the leak DROPS accuracy (that's the point).
        # Target is set below baseline so the agent can succeed after fixing.
        # For all other bugs: target is above baseline (agent must improve the pipeline).
        if self._ground_truth_bug == "data_leakage":
            # Expect ~10% accuracy drop after removing leaky column; target = honest floor
            target_accuracy = max(baseline.accuracy - 0.10, 0.75)
        else:
            # Target scales with difficulty: D1=+5%, D2=+8%, D3=+12%, D4=+15%
            increments = {1: 0.05, 2: 0.08, 3: 0.12, 4: 0.10}
            target_accuracy = min(baseline.accuracy + increments.get(difficulty, 0.10), 0.92)

        self._obs = Observation(
            done=False,
            reward=0.5,
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
        if self._obs is None:
            self.reset(difficulty=1)

        self._obs.step += 1
        result = self._execute_action(action)
        self._obs.last_action_result = result

        done, info = self._check_termination(action)
        self._obs.done = done
        self._obs.metadata = info

        # Use openenv rubric API: rubric(action, observation) -> float in (0, 1)
        self._obs.reward = self.rubric(action, self._obs)

        return self._obs

    @property
    def state(self) -> Observation:
        """Return current observation (read-only)."""
        if self._obs is None:
            self.reset(difficulty=1)
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
            canonical = self._BUG_ALIASES.get(action.bug_type, action.bug_type)
            if canonical != self._ground_truth_bug:
                self._wrong_submissions += 1
            return None
        return None

    # ------------------------------------------------------------------
    # ML evaluation (dispatches to sklearn or PyTorch based on pipeline config)
    # ------------------------------------------------------------------

    def _run_evaluation(self) -> ModelMetrics:
        if self._pipeline.get("model_type") == "pytorch":
            return self._run_pytorch_evaluation()
        return self._run_sklearn_evaluation()

    def _run_sklearn_evaluation(self) -> ModelMetrics:
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

    def _run_pytorch_evaluation(self) -> ModelMetrics:
        """Train a PyTorch MLP with the current pipeline config and return metrics."""
        try:
            import torch
            import torch.nn as nn
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            import numpy as np

            X = self._pipeline["X"]
            y = self._pipeline["y"]
            hp = self._pipeline.get("pytorch_hyperparams", {})

            lr = float(hp.get("learning_rate", 0.001))
            epochs = int(hp.get("epochs", 30))
            batch_size = int(hp.get("batch_size", 64))
            hidden_sizes = hp.get("hidden_sizes", [64, 32])
            activation_name = hp.get("activation", "relu")
            loss_fn_name = hp.get("loss_function", "crossentropy")
            optimizer_name = hp.get("optimizer", "adam")
            weight_decay = float(hp.get("weight_decay", 0.0))
            n_train_samples = hp.get("n_train_samples", None)

            X_arr = X.values.astype("float32") if hasattr(X, "values") else np.array(X, dtype="float32")
            y_arr = np.array(y, dtype="int64")

            X_train, X_val, y_train, y_val = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42
            )

            # Subsample training set if n_train_samples is set (overfitting bug)
            if n_train_samples is not None and n_train_samples < len(X_train):
                X_train = X_train[:int(n_train_samples)]
                y_train = y_train[:int(n_train_samples)]

            X_tr = torch.from_numpy(X_train)
            y_tr = torch.from_numpy(y_train)
            X_v = torch.from_numpy(X_val)
            y_v = torch.from_numpy(y_val)

            n_in = X_arr.shape[1]
            n_cls = len(np.unique(y_arr))

            # Choose activation class
            act_map = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
            act_cls = act_map.get(activation_name, nn.ReLU)

            dropout_rate = float(hp.get("dropout_rate", 0.0))

            # Build MLP with the configured activation and optional dropout
            layers = []
            prev = n_in
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), act_cls()]
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev = h
            layers.append(nn.Linear(prev, n_cls))
            model = nn.Sequential(*layers)

            # Apply weight initialization bug if flagged
            if self._pipeline.get("missing_weight_init"):
                def _bad_init(m):
                    if isinstance(m, nn.Linear):
                        nn.init.uniform_(m.weight, -10.0, 10.0)
                        nn.init.zeros_(m.bias)
                model.apply(_bad_init)

            if optimizer_name == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Choose loss function
            if loss_fn_name == "mse":
                criterion = nn.MSELoss()
                # MSE needs float targets in one-hot form
                y_tr_for_loss = torch.zeros(len(y_tr), n_cls).scatter_(
                    1, y_tr.unsqueeze(1), 1.0)
                y_v_for_loss = torch.zeros(len(y_v), n_cls).scatter_(
                    1, y_v.unsqueeze(1), 1.0)
            else:
                criterion = nn.CrossEntropyLoss()
                y_tr_for_loss = y_tr        # LongTensor for CrossEntropy
                y_v_for_loss = y_v

            # ── Training loop — track per-epoch loss ──────────────────
            n = len(X_tr)
            epoch_losses: list[float] = []
            diverged = False

            for _ in range(epochs):
                model.train()
                perm = torch.randperm(n)
                epoch_loss_sum, n_batches = 0.0, 0
                for start in range(0, n, batch_size):
                    idx = perm[start:start + batch_size]
                    optimizer.zero_grad()
                    out = model(X_tr[idx])
                    loss = criterion(out, y_tr_for_loss[idx] if loss_fn_name == "mse" else y_tr[idx])
                    if not torch.isfinite(loss):
                        diverged = True
                        break
                    loss.backward()
                    optimizer.step()
                    epoch_loss_sum += loss.item()
                    n_batches += 1
                if diverged:
                    epoch_losses.append(float("nan"))
                    break
                if n_batches > 0:
                    epoch_losses.append(round(epoch_loss_sum / n_batches, 5))

            # ── Gradient norm at final training state ─────────────────
            grad_norm = 0.0
            try:
                model.train()
                optimizer.zero_grad()
                sample_n = min(batch_size, n)
                s_out = model(X_tr[:sample_n])
                s_tgt = y_tr_for_loss[:sample_n] if loss_fn_name == "mse" else y_tr[:sample_n]
                s_loss = criterion(s_out, s_tgt)
                if torch.isfinite(s_loss):
                    s_loss.backward()
                    sq_sum = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    grad_norm = round(sq_sum ** 0.5, 8)
            except Exception:
                grad_norm = 0.0

            # ── Inference ──────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                train_out = model(X_tr)
                val_out = model(X_v)
                if not torch.isfinite(train_out).all():
                    # Diverged — return near-chance accuracy with the loss curve
                    n_cls_arr = len(np.unique(y_arr))
                    acc_chance = round(1.0 / max(n_cls_arr, 1), 4)
                    classes, counts = np.unique(y_arr, return_counts=True)
                    return ModelMetrics(
                        accuracy=acc_chance,
                        f1_score=acc_chance,
                        train_accuracy=acc_chance,
                        val_accuracy=acc_chance,
                        class_distribution={str(int(k)): int(v) for k, v in zip(classes, counts)},
                        loss_curve=epoch_losses,
                        gradient_norm=grad_norm,
                    )
                train_pred = train_out.argmax(dim=1).numpy()
                val_pred = val_out.argmax(dim=1).numpy()

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)
            classes, counts = np.unique(y_arr, return_counts=True)

            return ModelMetrics(
                accuracy=round(val_acc, 4),
                f1_score=round(val_f1, 4),
                train_accuracy=round(train_acc, 4),
                val_accuracy=round(val_acc, 4),
                class_distribution={str(int(k)): int(v) for k, v in zip(classes, counts)},
                loss_curve=epoch_losses,
                gradient_norm=grad_norm,
            )

        except ImportError:
            # torch not installed — graceful fallback to sklearn
            return self._run_sklearn_evaluation()

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    # Aliases: different names that mean the same bug
    _BUG_ALIASES = {
        "vanishing_gradients": "wrong_activation",
        "exploding_gradients": "wrong_learning_rate",
        "overfitting": "missing_regularization",
        "no_regularization": "missing_regularization",
        "wrong_dropout": "excessive_dropout",
        "too_much_dropout": "excessive_dropout",
    }

    def _check_termination(self, action: Action) -> tuple:
        submitted = action.bug_type if action.action_type == "submit_diagnosis" else None
        if submitted:
            # Normalise aliases
            canonical = self._BUG_ALIASES.get(submitted, submitted)
            if (
                canonical == self._ground_truth_bug
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
            # PyTorch diagnostics readable from run_code
            "loss_curve": self._obs.current_metrics.loss_curve if self._obs else [],
            "gradient_norm": self._obs.current_metrics.gradient_norm if self._obs else None,
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

            elif action.fix_type == "fix_learning_rate":
                lr = float(params.get("learning_rate", params.get("lr", 0.001)))
                self._pipeline.setdefault("pytorch_hyperparams", {})["learning_rate"] = lr
                return CodeExecutionResult(
                    stdout=f"Learning rate updated to {lr}. Run evaluate_model to retrain.",
                    stderr="", execution_time_ms=5,
                )

            elif action.fix_type == "fix_weight_initialization":
                self._pipeline["missing_weight_init"] = False
                return CodeExecutionResult(
                    stdout="Weight initialization fixed (PyTorch default Xavier/Kaiming).",
                    stderr="", execution_time_ms=5,
                )

            elif action.fix_type == "fix_batch_normalization":
                self._pipeline.setdefault("pytorch_hyperparams", {})["use_batch_norm"] = True
                return CodeExecutionResult(
                    stdout="Batch normalization layers enabled in the network.",
                    stderr="", execution_time_ms=5,
                )

            elif action.fix_type == "fix_activation_function":
                activation = params.get("activation", "relu")
                self._pipeline.setdefault("pytorch_hyperparams", {})["activation"] = activation
                return CodeExecutionResult(
                    stdout=f"Activation function updated to '{activation}'. Run evaluate_model to retrain.",
                    stderr="", execution_time_ms=5,
                )

            elif action.fix_type == "fix_loss_function":
                loss_fn = params.get("loss_function", "crossentropy")
                self._pipeline.setdefault("pytorch_hyperparams", {})["loss_function"] = loss_fn
                return CodeExecutionResult(
                    stdout=f"Loss function updated to '{loss_fn}'. Run evaluate_model to retrain.",
                    stderr="", execution_time_ms=5,
                )

            elif action.fix_type == "fix_dropout":
                rate = float(params.get("dropout_rate", 0.0))
                self._pipeline.setdefault("pytorch_hyperparams", {})["dropout_rate"] = rate
                return CodeExecutionResult(
                    stdout=f"Dropout rate updated to {rate}. Run evaluate_model to retrain.",
                    stderr="", execution_time_ms=5,
                )

            elif action.fix_type == "fix_regularization":
                wd = float(params.get("weight_decay", 0.001))
                hp = self._pipeline.setdefault("pytorch_hyperparams", {})
                hp["weight_decay"] = wd
                # Also remove the training-set size restriction so the model trains on full data
                if "n_train_samples" in hp:
                    del hp["n_train_samples"]
                return CodeExecutionResult(
                    stdout=f"Regularization applied: weight_decay={wd}, training on full dataset. Run evaluate_model to retrain.",
                    stderr="", execution_time_ms=5,
                )

        except Exception:
            return CodeExecutionResult(
                stdout="", stderr=traceback.format_exc(), execution_time_ms=0
            )

        return CodeExecutionResult(stdout="Fix applied.", stderr="", execution_time_ms=5)
