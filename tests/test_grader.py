# tests/test_grader.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from grader import MLDebuggerRubric
from actions import EvaluateModelAction, SubmitDiagnosisAction, InspectDataAction
from observations import ModelMetrics, Observation


def make_metrics(accuracy=0.7, train_accuracy=0.75):
    return ModelMetrics(
        accuracy=accuracy,
        f1_score=accuracy - 0.02,
        train_accuracy=train_accuracy,
        val_accuracy=accuracy,
        class_distribution={"0": 500, "1": 500},
    )


class FakeObs:
    """Minimal observation-like object for grader tests."""
    def __init__(self, baseline_acc=0.60, current_acc=0.75, target=0.80):
        self.baseline_metrics = make_metrics(baseline_acc)
        self.current_metrics = make_metrics(current_acc, train_accuracy=current_acc + 0.05)
        self.target_accuracy = target


class FakeState:
    """Mimics the env object passed to the grader."""
    def __init__(self, bug="data_leakage", baseline_acc=0.60, current_acc=0.75, target=0.80):
        self._ground_truth_bug = bug
        self._obs = FakeObs(baseline_acc, current_acc, target)


@pytest.fixture
def rubric():
    return MLDebuggerRubric()


def test_step_penalty_always_applied(rubric):
    # First inspect of a new target gets exploration bonus: -0.02 + 0.05 = +0.03
    state = FakeState()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    score = rubric.programmatic_score(action, state)
    assert score == pytest.approx(0.03, abs=1e-4)


def test_step_penalty_repeated_inspect(rubric):
    # Repeated inspect of same target: only step penalty -0.02
    state = FakeState()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    rubric.programmatic_score(action, state)  # first time
    score = rubric.programmatic_score(action, state)  # second time
    assert score == pytest.approx(-0.02, abs=1e-4)


def test_correct_diagnosis_bonus(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.85, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found leaky column.",
    )
    score = rubric.programmatic_score(action, state)
    # -0.02 + 0.3 + 0.5 = 0.78
    assert score == pytest.approx(0.78, abs=1e-4)


def test_wrong_diagnosis_penalty(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.50, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="class_imbalance",
        explanation="Wrong guess.",
    )
    score = rubric.programmatic_score(action, state)
    # -0.02 - 0.2 = -0.22
    assert score == pytest.approx(-0.22, abs=1e-4)


def test_accuracy_improvement_reward(rubric):
    state = FakeState(baseline_acc=0.60, current_acc=0.80)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric.programmatic_score(action, state)
    # delta = 0.20; delta*2 = 0.40; -0.02 step = 0.38
    assert score == pytest.approx(0.38, abs=1e-3)


def test_no_improvement_evaluate(rubric):
    state = FakeState(baseline_acc=0.70, current_acc=0.70)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric.programmatic_score(action, state)
    # delta = 0, step penalty only
    assert score == pytest.approx(-0.02, abs=1e-4)


def test_overfitting_penalty(rubric):
    state = FakeState(baseline_acc=0.60, current_acc=0.80)
    # Simulate large train/val gap
    state._obs.current_metrics = make_metrics(accuracy=0.80, train_accuracy=0.99)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric.programmatic_score(action, state)
    # delta*2 = 0.40, -0.02, -0.1 overfitting = 0.28
    assert score == pytest.approx(0.28, abs=1e-3)


def test_correct_diagnosis_but_accuracy_not_met(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.65, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found the bug.",
    )
    score = rubric.programmatic_score(action, state)
    # -0.02 + 0.3 = 0.28 (no +0.5 because accuracy not met)
    assert score == pytest.approx(0.28, abs=1e-4)


def test_llm_score_returns_zero_without_client(rubric):
    rubric.llm_available = False
    state = FakeState()
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found leaky column",
    )
    assert rubric.llm_score(action, state) == 0.0


def test_forward_combines_scores(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.85, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found leaky column.",
    )
    rubric.llm_available = False
    result = rubric.forward(action, state)
    assert result == pytest.approx(0.78, abs=1e-4)


def test_exploration_bonus_first_fix(rubric):
    state = FakeState()
    action = EvaluateModelAction(action_type="evaluate_model")
    # Trigger fix_attempted bonus via ApplyFixAction
    from actions import ApplyFixAction
    fix_action = ApplyFixAction(
        action_type="apply_fix",
        fix_type="resample_class_balance",
        parameters={},
    )
    score = rubric.programmatic_score(fix_action, state)
    # -0.02 + 0.05 first fix bonus = +0.03
    assert score == pytest.approx(0.03, abs=1e-4)


def test_exploration_bonus_not_repeated(rubric):
    state = FakeState()
    from actions import ApplyFixAction
    fix_action = ApplyFixAction(
        action_type="apply_fix",
        fix_type="resample_class_balance",
        parameters={},
    )
    rubric.programmatic_score(fix_action, state)  # first fix
    score = rubric.programmatic_score(fix_action, state)  # second fix
    assert score == pytest.approx(-0.02, abs=1e-4)  # no bonus


def test_reset_episode_clears_exploration(rubric):
    state = FakeState()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    rubric.programmatic_score(action, state)  # adds to seen
    rubric.reset_episode()
    score = rubric.programmatic_score(action, state)  # should get bonus again
    assert score == pytest.approx(0.03, abs=1e-4)
