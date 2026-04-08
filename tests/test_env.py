# tests/test_env.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import MLDebuggerEnv
from actions import (
    ApplyFixAction,
    EvaluateModelAction,
    InspectDataAction,
    RunCodeAction,
    SubmitDiagnosisAction,
)
from observations import Observation, ModelMetrics


@pytest.fixture
def env():
    e = MLDebuggerEnv()
    e.reset(difficulty=1)
    return e


def test_reset_returns_observation():
    e = MLDebuggerEnv()
    obs = e.reset(difficulty=1)
    assert isinstance(obs, Observation)
    assert obs.step == 0
    assert obs.max_steps == 15
    assert obs.task_description
    assert obs.episode_id
    assert obs.done is False


def test_step_increments_counter(env):
    action = InspectDataAction(action_type="inspect_data", target="shape")
    obs = env.step(action)
    assert obs.step == 1


def test_inspect_action_returns_output(env):
    action = InspectDataAction(action_type="inspect_data", target="describe")
    obs = env.step(action)
    assert obs.last_action_result is not None
    assert len(obs.last_action_result.stdout) > 0


def test_null_check_returns_output(env):
    action = InspectDataAction(action_type="inspect_data", target="null_check")
    obs = env.step(action)
    assert obs.last_action_result is not None
    assert obs.last_action_result.stderr == ""


def test_run_code_action(env):
    action = RunCodeAction(action_type="run_code", code="print(df.shape)")
    obs = env.step(action)
    assert obs.last_action_result is not None
    assert "(" in obs.last_action_result.stdout


def test_run_code_bad_code_returns_stderr(env):
    action = RunCodeAction(action_type="run_code", code="raise ValueError('test error')")
    obs = env.step(action)
    assert "ValueError" in obs.last_action_result.stderr


def test_evaluate_returns_metrics(env):
    action = EvaluateModelAction(action_type="evaluate_model")
    obs = env.step(action)
    assert isinstance(obs.current_metrics, ModelMetrics)
    assert 0.0 <= obs.current_metrics.accuracy <= 1.0


def test_evaluate_no_code_output(env):
    action = EvaluateModelAction(action_type="evaluate_model")
    obs = env.step(action)
    assert obs.last_action_result is None


def test_wrong_submission_increments(env):
    env._ground_truth_bug = "data_leakage"
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="class_imbalance",
        explanation="Wrong guess",
    )
    env.step(action)
    assert env._wrong_submissions == 1


def test_three_wrong_submissions_terminates(env):
    env._ground_truth_bug = "data_leakage"
    action = SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type="class_imbalance",
        explanation="Wrong",
    )
    obs = None
    for _ in range(3):
        obs = env.step(action)
    assert obs.done is True
    assert obs.metadata["reason"] == "too_many_wrong_submissions"


def test_timeout_terminates():
    e = MLDebuggerEnv()
    e.reset(difficulty=1)
    action = InspectDataAction(action_type="inspect_data", target="shape")
    obs = None
    for _ in range(15):
        obs = e.step(action)
        if obs.done:
            break
    assert obs.done is True
    assert obs.metadata.get("reason") == "timeout"


def test_all_difficulty_levels_reset():
    for d in [1, 2, 3, 4]:
        e = MLDebuggerEnv()
        obs = e.reset(difficulty=d)
        assert obs.step == 0
        assert 0.0 < obs.target_accuracy <= 1.0
        assert obs.max_steps == 15


def test_apply_fix_resample(env):
    action = ApplyFixAction(
        action_type="apply_fix",
        fix_type="resample_class_balance",
        parameters={"strategy": "oversample"},
    )
    obs = env.step(action)
    assert obs.last_action_result is not None
    assert obs.last_action_result.stderr == ""


def test_apply_fix_hyperparameter(env):
    action = ApplyFixAction(
        action_type="apply_fix",
        fix_type="set_hyperparameter",
        parameters={"key": "max_depth", "value": 10},
    )
    env.step(action)
    assert env._pipeline["hyperparams"].get("max_depth") == 10


def test_state_method_returns_observation(env):
    obs = env.state()
    assert isinstance(obs, Observation)


def test_reward_has_exploration_bonus_first_inspect(env):
    # First inspect of a new target: -0.02 + 0.05 exploration = +0.03
    action = InspectDataAction(action_type="inspect_data", target="shape")
    obs = env.step(action)
    assert obs.reward == pytest.approx(0.03, abs=1e-3)


def test_reward_is_negative_for_repeated_inspect(env):
    # Repeated inspect gives only step penalty
    action = InspectDataAction(action_type="inspect_data", target="shape")
    env.step(action)  # first call
    obs = env.step(action)  # second call
    assert obs.reward == pytest.approx(-0.02, abs=1e-3)


def test_obs_done_false_initially(env):
    obs = env.state()
    assert obs.done is False


def test_reset_with_seed_reproducible():
    e1, e2 = MLDebuggerEnv(), MLDebuggerEnv()
    obs1 = e1.reset(difficulty=1, seed=42)
    obs2 = e2.reset(difficulty=1, seed=42)
    assert obs1.episode_id != obs2.episode_id  # episode_id differs (kwarg not passed through)
    assert obs1.baseline_metrics.accuracy == obs2.baseline_metrics.accuracy
