#!/usr/bin/env python3
# demo.py
# Runs a heuristic demo agent and prints a summary table.
# Usage: python demo.py [--difficulty 1-4] [--episodes N] [--quiet]

import argparse
import random

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

# Smarter heuristic: tries different fixes based on inspection results
HEURISTIC_SEQUENCE = [
    lambda obs, env: InspectDataAction(action_type="inspect_data", target="describe"),
    lambda obs, env: InspectDataAction(action_type="inspect_data", target="value_counts", column="target"),
    lambda obs, env: RunCodeAction(
        action_type="run_code",
        code=(
            "import pandas as pd\n"
            "corr = df.corr()['target'].abs().sort_values(ascending=False)\n"
            "print('Top correlations with target:')\n"
            "print(corr.head(6))\n"
            "dist = df['target'].value_counts(normalize=True)\n"
            "print('\\nClass distribution (%):')\n"
            "print((dist * 100).round(1))"
        ),
    ),
    lambda obs, env: ApplyFixAction(
        action_type="apply_fix",
        fix_type="resample_class_balance",
        parameters={"strategy": "oversample"},
    ),
    lambda obs, env: EvaluateModelAction(action_type="evaluate_model"),
    lambda obs, env: ApplyFixAction(
        action_type="apply_fix",
        fix_type="set_hyperparameter",
        parameters={"key": "max_depth", "value": 10},
    ),
    lambda obs, env: EvaluateModelAction(action_type="evaluate_model"),
    lambda obs, env: SubmitDiagnosisAction(
        action_type="submit_diagnosis",
        bug_type=_guess_bug(obs, env),
        explanation=_build_explanation(obs, env),
    ),
]


def _guess_bug(obs, env):
    """Heuristic bug guess based on observations."""
    metrics = obs.current_metrics
    baseline = obs.baseline_metrics

    # Very high baseline accuracy suggests data leakage
    if baseline.accuracy > 0.97:
        return "data_leakage"
    # Low F1 relative to accuracy suggests class imbalance
    if baseline.accuracy - baseline.f1_score > 0.15:
        return "class_imbalance"
    # Very low train accuracy suggests wrong hyperparameter
    if baseline.train_accuracy < 0.65:
        return "wrong_hyperparameter"
    # Baseline close but slightly inflated → scaling error
    if 0.65 < baseline.accuracy < 0.80:
        return "scaling_error"
    return random.choice(BUG_TYPES)


def _build_explanation(obs, env):
    baseline = obs.baseline_metrics
    current = obs.current_metrics
    delta = current.accuracy - baseline.accuracy
    return (
        f"Baseline accuracy: {baseline.accuracy:.3f}. "
        f"After applying fixes, accuracy changed by {delta:+.3f} to {current.accuracy:.3f}. "
        f"Inspected feature correlations and class distribution. "
        f"Applied resampling and hyperparameter correction."
    )


def run_episode(difficulty=1, quiet=False):
    env = MLDebuggerEnv()
    obs = env.reset(difficulty=difficulty)

    if not quiet:
        print(f"\n{'='*65}")
        print(f"Difficulty: {difficulty}  |  Ground truth bug: {env._ground_truth_bug}")
        print(f"Task: {obs.task_description[:80]}")
        print(f"Baseline: {obs.baseline_metrics.accuracy:.4f}  |  Target: {obs.target_accuracy:.4f}")
        print(f"{'='*65}")

    total_reward = 0.0
    done = False
    step_log = []

    while not obs.done:
        idx = min(obs.step, len(HEURISTIC_SEQUENCE) - 1)
        action = HEURISTIC_SEQUENCE[idx](obs, env)
        obs = env.step(action)
        reward = obs.reward or 0.0
        total_reward += reward
        step_log.append((obs.step, action.action_type, reward, obs.current_metrics.accuracy, obs.done))

        if not quiet:
            out = ""
            if obs.last_action_result and obs.last_action_result.stdout:
                out = " | " + obs.last_action_result.stdout[:60].replace("\n", " ")
            print(
                f"  Step {obs.step:2d} | {action.action_type:<22s} | "
                f"reward={reward:+.3f} | acc={obs.current_metrics.accuracy:.4f}{out}"
            )

    if not quiet:
        print(f"\n  Result: {obs.metadata.get('reason')}  |  Total reward: {total_reward:+.3f}")

    return {
        "difficulty": difficulty,
        "bug": env._ground_truth_bug,
        "steps": obs.step,
        "total_reward": total_reward,
        "final_accuracy": obs.current_metrics.accuracy,
        "result": obs.metadata.get("reason"),
    }


def main():
    parser = argparse.ArgumentParser(description="ML Debugger Gym demo")
    parser.add_argument("--difficulty", type=int, default=0, help="1–4 (0 = run all)")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per difficulty level")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    difficulties = [args.difficulty] if args.difficulty > 0 else [1, 2, 3, 4]

    print("\nML Experiment Debugger Gym — Demo")
    print("=" * 65)

    summary = []
    for d in difficulties:
        for ep in range(args.episodes):
            result = run_episode(difficulty=d, quiet=args.quiet)
            summary.append(result)

    print("\n\nSummary")
    print("-" * 65)
    print(f"{'Diff':<6} {'Bug':<28} {'Steps':<7} {'Reward':<10} {'Accuracy':<10} {'Result'}")
    print("-" * 65)
    for r in summary:
        print(
            f"{r['difficulty']:<6} {r['bug']:<28} {r['steps']:<7} "
            f"{r['total_reward']:<+10.3f} {r['final_accuracy']:<10.4f} {r['result']}"
        )
    print("-" * 65)


if __name__ == "__main__":
    main()
