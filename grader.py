# grader.py
# MLDebuggerRubric — extends openenv.core Rubric for proper framework integration.

import json

from openenv.core.rubrics.base import Rubric


class MLDebuggerRubric(Rubric):
    """
    Scores agent actions. Extends openenv Rubric so the framework
    can call rubric(action, observation) and read rubric.last_score.

    Reward schedule (all values clamped to strict (0, 1)):
      base     0.01 per step
      +0.05    first time each inspect target is used
      +0.05    first apply_fix attempt in episode
      +0.3     correct bug type on submit_diagnosis
      +0.5     target accuracy reached at submission
      delta*2  accuracy improvement on evaluate_model
      -0.1     overfitting detected (train-val gap > 0.15)
    """

    def __init__(self):
        super().__init__()
        self._ground_truth_bug = None
        self._seen_inspect_targets: set = set()
        self._fix_attempted: bool = False

    # ── called by env at reset ──
    def set_ground_truth(self, bug_type: str):
        self._ground_truth_bug = bug_type

    def reset(self):
        """Called by Environment._reset_rubric()."""
        self._seen_inspect_targets = set()
        self._fix_attempted = False

    # ── openenv Rubric API ──
    def forward(self, action, observation) -> float:
        """Compute reward from action + resulting observation.

        Args:
            action:  one of the 5 Action subclasses
            observation:  Observation after the step

        Returns:
            float strictly in (0, 1)
        """
        score = 0.01  # small positive base (replaces negative step penalty)

        if action.action_type == "inspect_data":
            key = f"{action.target}:{getattr(action, 'column', '') or ''}"
            if key not in self._seen_inspect_targets:
                self._seen_inspect_targets.add(key)
                score += 0.05

        elif action.action_type == "apply_fix":
            if not self._fix_attempted:
                self._fix_attempted = True
                score += 0.05

        elif action.action_type == "evaluate_model":
            delta = observation.current_metrics.accuracy - observation.baseline_metrics.accuracy
            score += max(0.0, delta * 2.0)
            gap = observation.current_metrics.train_accuracy - observation.current_metrics.val_accuracy
            if gap > 0.15:
                score = max(score - 0.1, 0.01)

        elif action.action_type == "submit_diagnosis":
            # Accept both canonical name and common aliases
            _aliases = {"vanishing_gradients": "wrong_activation",
                        "exploding_gradients": "wrong_learning_rate",
                        "overfitting": "missing_regularization",
                        "no_regularization": "missing_regularization",
                        "wrong_dropout": "excessive_dropout",
                        "too_much_dropout": "excessive_dropout"}
            submitted = _aliases.get(action.bug_type, action.bug_type)
            if self._ground_truth_bug and submitted == self._ground_truth_bug:
                score += 0.3
            if observation.current_metrics.accuracy >= observation.target_accuracy:
                score += 0.5

        # Strict (0, 1) — never exactly 0.0 or 1.0
        return max(0.01, min(0.99, round(score, 4)))
