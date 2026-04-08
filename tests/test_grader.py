# tests/test_grader.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from grader import MLDebuggerRubric
from actions import EvaluateModelAction, SubmitDiagnosisAction, InspectDataAction, ApplyFixAction
from observations import ModelMetrics, Observation


def make_metrics(accuracy=0.7, train_accuracy=0.75):
    return ModelMetrics(
        accuracy=accuracy,
        f1_score=accuracy - 0.02,
        train_accuracy=train_accuracy,
        val_accuracy=accuracy,
        class_distribution={"0": 500, "1": 500},
    )


def make_obs(baseline_acc=0.60, current_acc=0.75, target=0.80):
    return Observation(
        done=False,
        reward=0.5,
        step=1,
        max_steps=15,
        task_description="test",
        current_metrics=make_metrics(current_acc, current_acc + 0.05),
        baseline_metrics=make_metrics(baseline_acc),
        target_accuracy=target,
        episode_id="test",
    )


@pytest.fixture
def rubric():
    r = MLDebuggerRubric()
    r.set_ground_truth("data_leakage")
    return r


def test_step_base_score(rubric):
    obs = make_obs()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    # First inspect of new target: 0.01 + 0.05 = 0.06
    score = rubric(action, obs)
    assert score == pytest.approx(0.06, abs=1e-3)


def test_repeated_inspect_no_bonus(rubric):
    obs = make_obs()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    rubric(action, obs)  # first time
    score = rubric(action, obs)  # second time
    assert score == pytest.approx(0.01, abs=1e-3)


def test_correct_diagnosis_bonus(rubric):
    obs = make_obs(current_acc=0.85, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found leaky column.",
    )
    score = rubric(action, obs)
    # 0.01 + 0.3 + 0.5 = 0.81
    assert score == pytest.approx(0.81, abs=1e-3)


def test_wrong_diagnosis_base_only(rubric):
    obs = make_obs(current_acc=0.50, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="class_imbalance",
        explanation="Wrong guess.",
    )
    score = rubric(action, obs)
    # 0.01 (no bonus for wrong guess, no target met) → clamped to 0.01
    assert score == pytest.approx(0.01, abs=1e-3)


def test_accuracy_improvement_reward(rubric):
    obs = make_obs(baseline_acc=0.60, current_acc=0.80)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric(action, obs)
    # 0.01 + delta*2 = 0.01 + 0.40 = 0.41
    assert score == pytest.approx(0.41, abs=1e-2)


def test_no_improvement_evaluate(rubric):
    obs = make_obs(baseline_acc=0.70, current_acc=0.70)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric(action, obs)
    assert score == pytest.approx(0.01, abs=1e-3)


def test_overfitting_penalty(rubric):
    obs = make_obs(baseline_acc=0.60, current_acc=0.80)
    obs.current_metrics = make_metrics(accuracy=0.80, train_accuracy=0.99)
    action = EvaluateModelAction(action_type="evaluate_model")
    score = rubric(action, obs)
    # 0.01 + 0.40 - 0.1 = 0.31
    assert score == pytest.approx(0.31, abs=1e-2)


def test_correct_diagnosis_but_accuracy_not_met(rubric):
    obs = make_obs(current_acc=0.65, target=0.80)
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="data_leakage",
        explanation="Found the bug.",
    )
    score = rubric(action, obs)
    # 0.01 + 0.3 = 0.31 (no +0.5 because accuracy not met)
    assert score == pytest.approx(0.31, abs=1e-3)


def test_exploration_bonus_first_fix(rubric):
    obs = make_obs()
    fix_action = ApplyFixAction(
        action_type="apply_fix",
        fix_type="resample_class_balance",
        parameters={},
    )
    score = rubric(fix_action, obs)
    # 0.01 + 0.05 first fix bonus = 0.06
    assert score == pytest.approx(0.06, abs=1e-3)


def test_exploration_bonus_not_repeated(rubric):
    obs = make_obs()
    fix_action = ApplyFixAction(
        action_type="apply_fix",
        fix_type="resample_class_balance",
        parameters={},
    )
    rubric(fix_action, obs)  # first fix
    score = rubric(fix_action, obs)  # second fix
    assert score == pytest.approx(0.01, abs=1e-3)


def test_reset_clears_exploration(rubric):
    obs = make_obs()
    action = InspectDataAction(action_type="inspect_data", target="shape")
    rubric(action, obs)  # adds to seen
    rubric.reset()
    score = rubric(action, obs)  # should get bonus again
    assert score == pytest.approx(0.06, abs=1e-3)


def test_all_scores_strictly_in_range(rubric):
    """Every possible score must be in (0, 1) exclusive."""
    obs = make_obs()
    actions = [
        InspectDataAction(action_type="inspect_data", target="shape"),
        EvaluateModelAction(action_type="evaluate_model"),
        ApplyFixAction(action_type="apply_fix", fix_type="resample_class_balance", parameters={}),
        SubmitDiagnosisAction(action_type="submit_diagnosis", bug_type="data_leakage", explanation="test"),
        SubmitDiagnosisAction(action_type="submit_diagnosis", bug_type="wrong_hyperparameter", explanation="test"),
    ]
    for a in actions:
        score = rubric(a, obs)
        assert 0 < score < 1, f"Score {score} out of (0,1) for {a.action_type}"


def test_last_score_set(rubric):
    """openenv Rubric stores last_score after __call__."""
    obs = make_obs()
    action = InspectDataAction(action_type="inspect_data", target="describe")
    result = rubric(action, obs)
    assert rubric.last_score == result
