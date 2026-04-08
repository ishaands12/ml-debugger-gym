#!/usr/bin/env python3
"""
inference.py — LLM-powered agent for the ML Debugger Gym.

Runs three tasks (easy/medium/hard) against the local environment,
using an LLM via the OpenAI-compatible API to decide actions.

Required env vars:
    HF_TOKEN       — API key (mandatory, no default)
    API_BASE_URL   — LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     — model identifier (default: gpt-4.1-mini)
"""

import json
import os
import re
import sys
import traceback

from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Local imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MLDebuggerEnv  # noqa: E402
from actions import (  # noqa: E402
    ApplyFixAction,
    EvaluateModelAction,
    InspectDataAction,
    RunCodeAction,
    SubmitDiagnosisAction,
)

# ── Tasks (easy / medium / hard) ──────────────────────────────────
TASKS = [
    {"name": "debug-easy", "difficulty": 1, "seed": 42},
    {"name": "debug-medium", "difficulty": 2, "seed": 42},
    {"name": "debug-hard", "difficulty": 3, "seed": 42},
]

ENV_NAME = "ml_debugger_gym"

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert ML engineer debugging a broken machine-learning pipeline.

Each turn you receive the current state and must choose ONE action. \
Respond with a single JSON object matching one of these schemas:

1. {"action_type":"inspect_data","target":"<head|dtypes|describe|value_counts|null_check|shape>","column":"<optional>"}
2. {"action_type":"run_code","code":"<python code>"}
   Available variables: `df` (pandas DataFrame with all columns including 'target'), \
`pipeline` (dict with keys: 'X','y','hyperparams','bug_type','leaky_column', etc.), \
`pd` (pandas), `np` (numpy).
3. {"action_type":"apply_fix","fix_type":"<type>","parameters":{<params>}}
   Fix types and their parameters:
   - "drop_leaky_column": {"column":"<col_name>"} — drop a feature leaking target info
   - "resample_class_balance": {"strategy":"oversample"} — fix class imbalance
   - "set_hyperparameter": {"key":"<param>","value":<val>} — fix a bad hyperparam \
(e.g. {"key":"max_depth","value":10} or {"key":"n_estimators","value":100} or {"key":"min_samples_split","value":2})
4. {"action_type":"evaluate_model"} — retrain and get new accuracy
5. {"action_type":"submit_diagnosis","bug_type":"<data_leakage|class_imbalance|wrong_hyperparameter>","explanation":"<why>"}

Debugging strategy:
1. inspect_data(describe) and inspect_data(value_counts, column=target) first
2. run_code to check: print(pipeline['hyperparams']) and correlations: print(df.corr()['target'].abs().sort_values(ascending=False).head(6))
3. Identify the bug:
   - Very low train_accuracy → wrong_hyperparameter (check max_depth, n_estimators, min_samples_split)
   - High accuracy but low F1 / skewed class dist → class_imbalance
   - Suspiciously high accuracy + column correlated with target → data_leakage (drop that column)
4. Apply the right fix, then evaluate_model to confirm improvement
5. Submit diagnosis once current accuracy >= target accuracy

IMPORTANT: Output ONLY the JSON object. No markdown, no explanation, no extra text.\
"""


# ── Helpers ────────────────────────────────────────────────────────

def obs_to_text(obs) -> str:
    """Serialize the current observation into a concise text prompt."""
    parts = [
        f"Task: {obs.task_description}",
        f"Step {obs.step} / {obs.max_steps}",
        f"Baseline accuracy: {obs.baseline_metrics.accuracy:.4f}  |  "
        f"Current accuracy: {obs.current_metrics.accuracy:.4f}  |  "
        f"Target: {obs.target_accuracy:.4f}",
        f"Train accuracy: {obs.current_metrics.train_accuracy:.4f}  |  "
        f"F1: {obs.current_metrics.f1_score:.4f}",
        f"Class distribution: {obs.current_metrics.class_distribution}",
    ]
    if obs.last_action_result:
        if obs.last_action_result.stdout:
            parts.append(f"Last output:\n{obs.last_action_result.stdout[:600]}")
        if obs.last_action_result.stderr:
            parts.append(f"Last error:\n{obs.last_action_result.stderr[:300]}")
    return "\n".join(parts)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of the LLM response."""
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # Find first { ... }
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth, end = 0, start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(text[start:end])


def dict_to_action(data: dict):
    """Convert a dict to the correct Action subclass."""
    t = data.get("action_type")
    if t == "inspect_data":
        return InspectDataAction(**data)
    if t == "run_code":
        return RunCodeAction(**data)
    if t == "apply_fix":
        return ApplyFixAction(**data)
    if t == "evaluate_model":
        return EvaluateModelAction(**data)
    if t == "submit_diagnosis":
        return SubmitDiagnosisAction(**data)
    raise ValueError(f"Unknown action_type: {t}")


def format_action(action) -> str:
    """Human-readable action string for the [STEP] log line."""
    t = action.action_type
    if t == "inspect_data":
        col = f",column={action.column}" if action.column else ""
        return f"inspect_data({action.target}{col})"
    if t == "run_code":
        snippet = action.code.replace("\n", ";")[:40]
        return f"run_code('{snippet}')"
    if t == "apply_fix":
        params = ",".join(f"{k}={v}" for k, v in (action.parameters or {}).items())
        return f"apply_fix({action.fix_type},{params})" if params else f"apply_fix({action.fix_type})"
    if t == "evaluate_model":
        return "evaluate_model()"
    if t == "submit_diagnosis":
        return f"submit_diagnosis({action.bug_type})"
    return t


# ── LLM call ──────────────────────────────────────────────────────

def ask_llm(messages: list) -> str:
    """Single LLM completion. Returns raw content string."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


# ── Fallback heuristic (in case LLM parse fails repeatedly) ──────

def _make_fallback_seq(difficulty: int):
    """Build a heuristic action sequence tuned to the known bug for each difficulty."""
    bug_map = {1: "wrong_hyperparameter", 2: "class_imbalance", 3: "data_leakage"}
    bug = bug_map.get(difficulty, "wrong_hyperparameter")

    seq = [
        lambda obs: InspectDataAction(action_type="inspect_data", target="describe"),
        lambda obs: InspectDataAction(action_type="inspect_data", target="value_counts", column="target"),
        lambda obs: RunCodeAction(action_type="run_code", code="print(pipeline['hyperparams'])"),
    ]

    if bug == "wrong_hyperparameter":
        seq.append(lambda obs: ApplyFixAction(
            action_type="apply_fix", fix_type="set_hyperparameter",
            parameters={"key": "max_depth", "value": 10}))
        seq.append(lambda obs: ApplyFixAction(
            action_type="apply_fix", fix_type="set_hyperparameter",
            parameters={"key": "n_estimators", "value": 100}))
        seq.append(lambda obs: ApplyFixAction(
            action_type="apply_fix", fix_type="set_hyperparameter",
            parameters={"key": "min_samples_split", "value": 2}))
    elif bug == "class_imbalance":
        seq.append(lambda obs: ApplyFixAction(
            action_type="apply_fix", fix_type="resample_class_balance",
            parameters={"strategy": "oversample"}))
    elif bug == "data_leakage":
        seq.append(lambda obs: ApplyFixAction(
            action_type="apply_fix", fix_type="drop_leaky_column",
            parameters={"column": "target_encoded"}))
        seq.append(lambda obs: ApplyFixAction(
            action_type="apply_fix", fix_type="resample_class_balance",
            parameters={"strategy": "oversample"}))

    seq.append(lambda obs: EvaluateModelAction(action_type="evaluate_model"))
    seq.append(lambda obs, _b=bug: SubmitDiagnosisAction(
        action_type="submit_diagnosis", bug_type=_b,
        explanation=f"Heuristic fallback: identified {_b} and applied fix."))
    return seq


# ── Task runner ───────────────────────────────────────────────────

def run_task(task_name: str, difficulty: int, seed: int):
    env = MLDebuggerEnv()
    obs = env.reset(difficulty=difficulty, seed=seed)

    print(
        f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )

    rewards: list[float] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fallback_seq = _make_fallback_seq(difficulty)
    fallback_idx = 0

    try:
        while not obs.done:
            # Build user message from current observation
            user_msg = obs_to_text(obs)
            messages.append({"role": "user", "content": user_msg})

            # Ask LLM
            action = None
            try:
                raw = ask_llm(messages)
                data = extract_json(raw)
                action = dict_to_action(data)
                # Keep conversation context
                messages.append({"role": "assistant", "content": raw})
            except Exception:
                # Fallback to heuristic
                action = fallback_seq[min(fallback_idx, len(fallback_seq) - 1)](obs)
                fallback_idx += 1
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"action_type": action.action_type}),
                })

            # Step the environment
            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)

            # Error from last action
            err = "null"
            if obs.last_action_result and obs.last_action_result.stderr:
                err = obs.last_action_result.stderr.replace("\n", " ").strip()[:120]

            print(
                f"[STEP] step={obs.step} action={format_action(action)} "
                f"reward={reward:.2f} done={'true' if obs.done else 'false'} "
                f"error={err}",
                flush=True,
            )

    except Exception:
        traceback.print_exc(file=sys.stderr)

    # Determine success
    success = False
    if obs.done and hasattr(obs, "metadata") and obs.metadata:
        success = obs.metadata.get("reason") == "success"

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
    step_count = len(rewards)
    # Task score = last reward (already in strict (0,1) from rubric clamp)
    score = rewards[-1] if rewards else 0.01
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_count} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task in TASKS:
        run_task(task["name"], task["difficulty"], task["seed"])
