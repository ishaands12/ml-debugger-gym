---
title: ML Debugger Gym Environment Server
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - machine-learning
  - pytorch
---

# ML Experiment Debugger Gym

An RL environment where an agent acts as a data scientist debugging broken ML pipelines —
including **PyTorch neural networks** and scikit-learn classifiers.

Built for the **Meta PyTorch OpenEnv Hackathon** (Bangalore, April 25–26, 2026).

---

## Environment Overview

Each episode gives the agent a broken ML pipeline — a dataset and model that runs without
crashing but produces wrong results due to a silently injected bug. The agent must:

1. **Inspect** the data and pipeline to form a hypothesis
2. **Apply a fix** targeting the suspected bug
3. **Evaluate** the model to verify the fix improved accuracy
4. **Submit a diagnosis** naming the bug and explaining the reasoning

### Bug categories

| Bug | Description |
|-----|-------------|
| `wrong_hyperparameter` | Bad sklearn RF param (e.g. `min_samples_split=400`) causes underfitting |
| `class_imbalance` | 92% majority class — accuracy looks fine but F1 is broken |
| `data_leakage` | A `target_encoded` column leaks the label into features |
| `wrong_learning_rate` | **PyTorch MLP** with `lr=50.0` — gradients diverge, near-chance accuracy |

Four difficulty levels progress from obvious single bugs (D1) to multi-bug PyTorch scenarios (D5).

---

## Quick Start

```bash
git clone https://github.com/ishaanb10/ml-debugger-gym
cd ml-debugger-gym
pip install -r requirements.txt

# Sanity check — heuristic agent through all 4 difficulty levels
python test_agent.py

# Full test suite
python -m pytest tests/ -v
```

---

## Action Space

| Action | `action_type` | Key Parameters | Purpose |
|--------|---------------|----------------|---------|
| `InspectDataAction` | `inspect_data` | `target` (head/dtypes/describe/value_counts/null_check/shape), `column` | Non-destructive data inspection |
| `RunCodeAction` | `run_code` | `code: str` | Sandboxed Python execution (pandas/numpy/torch available) |
| `ApplyFixAction` | `apply_fix` | `fix_type` (enum), `parameters: dict` | Applies a structured fix to the pipeline |
| `EvaluateModelAction` | `evaluate_model` | — | Retrains model (sklearn or PyTorch) and returns updated metrics |
| `SubmitDiagnosisAction` | `submit_diagnosis` | `bug_type` (enum), `explanation: str` | Formal diagnosis — terminates episode on success |

### Apply-fix types

| `fix_type` | Parameters | Use case |
|---|---|---|
| `set_hyperparameter` | `{"key": "min_samples_split", "value": 2}` | Fix bad sklearn RF hyperparameter |
| `resample_class_balance` | `{"strategy": "oversample"}` | Fix class imbalance |
| `drop_leaky_column` | `{"column": "target_encoded"}` | Remove leaky feature |
| `fix_learning_rate` | `{"learning_rate": 0.001}` | Fix bad PyTorch optimizer LR |
| `fix_weight_initialization` | `{}` | Reset PyTorch weights to Kaiming default |
| `fix_batch_normalization` | `{}` | Enable BatchNorm layers in PyTorch MLP |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Episode step budget (always 15) |
| `task_description` | `str` | Natural language hint about the bug |
| `current_metrics` | `ModelMetrics` | Accuracy/F1 after last `evaluate_model` |
| `baseline_metrics` | `ModelMetrics` | Metrics at episode start (frozen) |
| `target_accuracy` | `float` | Accuracy threshold to hit before submitting |
| `last_action_result` | `CodeExecutionResult` | stdout/stderr from last code action |
| `available_actions` | `List[str]` | Valid action types at this step |
| `episode_id` | `str` | UUID for logging and reproducibility |

The observation is also injected with `model_type` in the text prompt so the LLM agent can
immediately decide the right debugging strategy.

---

## Reward Structure

| Event | Reward | Condition |
|-------|--------|-----------|
| Base step | +0.01 | Every step |
| Exploration bonus | +0.05 | First use of each inspect target / first fix attempt |
| Accuracy improvement | `delta × 2.0` | Positive accuracy delta on `evaluate_model` |
| Overfitting penalty | −0.10 (floor 0.01) | `train_acc − val_acc > 0.15` on evaluate |
| Correct bug type | +0.30 | `submit_diagnosis.bug_type == ground_truth` |
| Target accuracy reached | +0.50 | `accuracy >= target_accuracy` at submission |

All rewards are clamped to **(0.01, 0.99)** — never exactly 0 or 1 as required by openenv.

---

## Difficulty Levels

| Level | Label | Model | Bugs | Hints |
|-------|-------|-------|------|-------|
| 1 | Easy | sklearn RF | 1 (obvious) | Full — names the symptom clearly |
| 2 | Medium | sklearn RF | 1 (subtle) | Partial — bug category mentioned |
| 3 | Hard | sklearn RF | 2 (mixed) | Minimal — only bug count |
| 4 | Expert | sklearn RF | 2 + obfuscated columns | None |
| 5 | PyTorch | PyTorch MLP | 1 (bad LR) | Hints to check optimizer config |

---

## Inference Script

`inference.py` uses the OpenAI-compatible API to run an LLM agent through all four tasks:

| Task | Difficulty | Model | Bug |
|------|-----------|-------|-----|
| debug-easy | 1 | sklearn RF | wrong_hyperparameter |
| debug-medium | 2 | sklearn RF | class_imbalance |
| debug-hard | 3 | sklearn RF | data_leakage |
| debug-pytorch | 5 | PyTorch MLP | wrong_learning_rate |

```bash
HF_TOKEN=<your-key> python inference.py
```

| Env Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | Yes (used as API key) |
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4.1-mini` | No |

Output follows the `[START]` / `[STEP]` / `[END]` format specified by the hackathon submission guidelines.

---

## PyTorch MLP Architecture

For difficulty 5 (`wrong_learning_rate`), the environment trains a simple feedforward MLP:

```
Input(n_features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(n_classes)
```

Trained with `CrossEntropyLoss` and `Adam`. The bug injects `learning_rate=50.0` which causes
immediate gradient divergence. After `fix_learning_rate(learning_rate=0.001)`, the model
converges to ~85% accuracy in 30 epochs.

---

## File Structure

```
ml-debugger-gym/
├── inference.py            # LLM agent (submission entry point)
├── env.py                  # MLDebuggerEnv (OpenEnv BaseEnvironment)
├── actions.py              # 5 Pydantic Action models
├── observations.py         # Observation + ModelMetrics models
├── grader.py               # MLDebuggerRubric (extends openenv Rubric)
├── dataset_generator.py    # Procedural broken pipeline generation
├── client.py               # OpenEnv HTTP client
├── server/app.py           # FastAPI server (openenv create_app)
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # python:3.11-slim + torch CPU + openenv deps
├── requirements.txt
├── test_agent.py           # Heuristic test agent (sanity check)
├── demo.py                 # Demo script with argparse CLI
└── tests/
    ├── test_env.py
    └── test_grader.py
```

---

## License

MIT
