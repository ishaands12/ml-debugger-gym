# test_agent.py
# Heuristic agent that exercises every action type.
# Run to verify the environment works end-to-end before submitting.

import random
import argparse

from env import MLDebuggerEnv
from actions import (
    ApplyFixAction,
    EvaluateModelAction,
    InspectDataAction,
    RunCodeAction,
    SubmitDiagnosisAction,
)

BUG_TYPES = [
    "data_leakage",
    "class_imbalance",
    "wrong_hyperparameter",
    "scaling_error",
    "train_test_contamination",
    "missing_value_handling",
]


def heuristic_action(obs, step):
    if step == 0:
        return InspectDataAction(action_type="inspect_data", target="describe")
    elif step == 1:
        return InspectDataAction(action_type="inspect_data", target="null_check")
    elif step == 2:
        return InspectDataAction(action_type="inspect_data", target="value_counts", column="target")
    elif step == 3:
        return RunCodeAction(
            action_type="run_code",
            code=(
                "import pandas as pd\n"
                "corr = df.corr()['target'].abs().sort_values(ascending=False)\n"
                "print(corr.head(5))"
            ),
        )
    elif step == 4:
        return ApplyFixAction(
            action_type="apply_fix",
            fix_type="resample_class_balance",
            parameters={"strategy": "oversample"},
        )
    elif step == 5:
        return EvaluateModelAction(action_type="evaluate_model")
    else:
        return SubmitDiagnosisAction(
            action_type="submit_diagnosis",
            bug_type=random.choice(BUG_TYPES),
            explanation="Inspected data and found issues. Applied resampling and re-evaluated.",
        )


def run_episode(difficulty: int = 1, quiet: bool = False) -> float:
    env = MLDebuggerEnv()
    obs = env.reset(difficulty=difficulty)

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Episode | Difficulty: {difficulty} | Bug: {env._ground_truth_bug}")
        print(f"Task: {obs.task_description}")
        print(f"Baseline: {obs.baseline_metrics.accuracy:.4f}  Target: {obs.target_accuracy:.4f}")
        print(f"{'='*60}")

    total_reward = 0.0

    while not obs.done:
        action = heuristic_action(obs, obs.step)
        obs = env.step(action)
        reward = obs.reward or 0.0
        total_reward += reward

        if not quiet:
            acc = obs.current_metrics.accuracy
            print(
                f"  Step {obs.step:2d} | {action.action_type:<20s} | "
                f"reward={reward:+.3f} | acc={acc:.4f} | done={obs.done}"
            )
            if obs.last_action_result and obs.last_action_result.stdout:
                preview = obs.last_action_result.stdout[:120].replace("\n", " ")
                print(f"           output: {preview}")
            if obs.last_action_result and obs.last_action_result.stderr:
                print(f"           ERROR:  {obs.last_action_result.stderr[:120]}")

    if not quiet:
        reason = obs.metadata.get("reason", "unknown")
        print(f"\nResult: {reason} | Total reward: {total_reward:+.3f}")

    return total_reward


def main():
    parser = argparse.ArgumentParser(description="ML Debugger Gym heuristic test agent")
    parser.add_argument("--difficulty", type=int, default=0, help="1-4 (0 = run all)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    difficulties = [args.difficulty] if args.difficulty > 0 else [1, 2, 3, 4]

    for d in difficulties:
        rewards = [run_episode(difficulty=d, quiet=args.quiet) for _ in range(args.episodes)]
        print(f"\nDifficulty {d}: avg reward = {sum(rewards)/len(rewards):+.3f}")


if __name__ == "__main__":
    main()
