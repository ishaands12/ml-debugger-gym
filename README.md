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

| Bug | Model | Description |
|-----|-------|-------------|
| `wrong_hyperparameter` | sklearn RF | Bad param (e.g. `min_samples_split=400`) causes severe underfitting |
| `class_imbalance` | sklearn RF | 92% majority class — accuracy looks fine but F1 is broken |
| `data_leakage` | sklearn RF | A `target_encoded` column leaks the label into features |
| `wrong_learning_rate` | **PyTorch MLP** | `lr=50.0` → gradients diverge immediately, near-chance accuracy |
| `wrong_activation` | **PyTorch MLP** | 4-layer sigmoid network → vanishing gradients, ~50-65% accuracy |
| `wrong_loss_function` | **PyTorch MLP** | MSE loss on one-hot targets instead of CrossEntropy → weak gradient signal |

Seven difficulty levels — D1–D4 are sklearn, D5–D7 are PyTorch-native.

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

| Level | Label | Model | Bug | Hint |
|-------|-------|-------|-----|------|
| 1 | Easy | sklearn RF | wrong_hyperparameter | Full — names the symptom |
| 2 | Medium | sklearn RF | class_imbalance | Partial |
| 3 | Hard | sklearn RF | data_leakage + 1 secondary | Minimal |
| 4 | Expert | sklearn RF | wrong_hyperparameter + obfuscated cols | None |
| 5 | PyTorch-LR | PyTorch MLP | wrong_learning_rate (lr=50) | Check optimizer config |
| 6 | PyTorch-Activation | PyTorch MLP (4-layer) | wrong_activation (sigmoid) | Check activation function |
| 7 | PyTorch-Loss | PyTorch MLP | wrong_loss_function (MSE) | Check loss function |

---

## Inference Script

`inference.py` uses the OpenAI-compatible API to run an LLM agent through all four tasks.
**3 of 4 tasks use PyTorch** — directly relevant to the hackathon theme.

| Task | Difficulty | Model | Bug |
|------|-----------|-------|-----|
| debug-sklearn | 1 | sklearn RF | wrong_hyperparameter (`min_samples_split=400`) |
| debug-pytorch-vanishing | 6 | PyTorch MLP | wrong_activation (sigmoid → vanishing gradients) |
| debug-pytorch-loss | 7 | PyTorch MLP | wrong_loss_function (MSE instead of CrossEntropy) |
| debug-pytorch-lr | 5 | PyTorch MLP | wrong_learning_rate (`lr=50.0` → divergence) |

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
