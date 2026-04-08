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
    state = FakeState()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    score = rubric.programmatic_score(action, state)
    assert score == pytest.approx(-0.05, abs=1e-4)


def test_correct_diagnosis_bonus(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.85, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found leaky column.",
    )
    score = rubric.programmatic_score(action, state)
    # -0.05 + 0.3 + 0.5 = 0.75
    assert score == pytest.approx(0.75, abs=1e-4)


def test_wrong_diagnosis_penalty(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.50, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="class_imbalance",
        explanation="Wrong guess.",
    )
    score = rubric.programmatic_score(action, state)
    # -0.05 - 0.2 = -0.25
    assert score == pytest.approx(-0.25, abs=1e-4)


def test_accuracy_improvement_reward(rubric):
    state = FakeState(baseline_acc=0.60, current_acc=0.80)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric.programmatic_score(action, state)
    # delta = 0.20; delta*2 = 0.40; -0.05 step = 0.35
    assert score == pytest.approx(0.35, abs=1e-3)


def test_no_improvement_evaluate(rubric):
    state = FakeState(baseline_acc=0.70, current_acc=0.70)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric.programmatic_score(action, state)
    # delta = 0, step penalty only
    assert score == pytest.approx(-0.05, abs=1e-4)


def test_overfitting_penalty(rubric):
    state = FakeState(baseline_acc=0.60, current_acc=0.80)
    # Simulate large train/val gap
    state._obs.current_metrics = make_metrics(accuracy=0.80, train_accuracy=0.99)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric.programmatic_score(action, state)
    # delta*2 = 0.40, -0.05, -0.1 overfitting = 0.25
    assert score == pytest.approx(0.25, abs=1e-3)


def test_correct_diagnosis_but_accuracy_not_met(rubric):
    state = FakeState(bug="data_leakage", current_acc=0.65, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found the bug.",
    )
    score = rubric.programmatic_score(action, state)
    # -0.05 + 0.3 = 0.25 (no +0.5 because accuracy not met)
    assert score == pytest.approx(0.25, abs=1e-4)


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
    # LLM unavailable in test; forward = programmatic only
    rubric.llm_available = False
    result = rubric.forward(action, state)
    assert result == pytest.approx(0.75, abs=1e-4)
