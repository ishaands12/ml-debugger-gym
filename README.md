# ML Experiment Debugger Gym

An RL environment where an agent acts as a data scientist debugging broken ML pipelines.

Built for the **Meta PyTorch OpenEnv Hackathon** (Bangalore, April 25–26, 2026).

---

## Environment Overview

Each episode gives the agent a broken ML pipeline — a dataset and training script that runs without crashing but produces wrong results due to a silently injected bug. The agent must:

1. **Inspect** the data and pipeline to form a hypothesis
2. **Apply a fix** targeting the suspected bug
3. **Evaluate** the model to verify the fix improved accuracy
4. **Submit a diagnosis** naming the bug and explaining the reasoning

Six bug categories are injected: data leakage, class imbalance, wrong hyperparameters, scaling errors, train/test contamination, and missing value handling errors. Four difficulty levels progress from obvious single bugs to adversarial multi-bug scenarios.

---

## Quick Start

```bash
git clone https://github.com/yourusername/ml-debugger-gym
cd ml-debugger-gym
pip install -r requirements.txt

# Sanity check — runs heuristic agent through all 4 difficulty levels
python test_agent.py

# Full test suite
python -m pytest tests/ -v
```

---

## Action Space

| Action | `action_type` | Key Parameters | Purpose |
|--------|---------------|----------------|---------|
| `InspectDataAction` | `inspect_data` | `target` (head/dtypes/describe/value_counts/null_check/shape), `column` | Non-destructive data inspection |
| `RunCodeAction` | `run_code` | `code: str` | Sandboxed Python execution (pandas/sklearn) |
| `ApplyFixAction` | `apply_fix` | `fix_type` (enum), `parameters: dict` | Applies a structured fix to the pipeline |
| `EvaluateModelAction` | `evaluate_model` | — | Runs full train/eval cycle, returns updated metrics |
| `SubmitDiagnosisAction` | `submit_diagnosis` | `bug_type` (enum), `explanation: str` | Formal diagnosis — terminates episode on success |

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

---

## Reward Structure

| Event | Reward | Condition |
|-------|--------|-----------|
| Step penalty | −0.05 | Every step, always |
| Correct bug type | +0.30 | `submit_diagnosis.bug_type == ground_truth` |
| Wrong bug type | −0.20 | `submit_diagnosis.bug_type != ground_truth` |
| Target accuracy reached | +0.50 | `accuracy >= target_accuracy` at submission |
| Accuracy improvement | `delta × 2.0` | Positive delta on `evaluate_model` |
| Overfitting penalty | −0.10 | `train_acc − val_acc > 0.15` on evaluate |
| Timeout | −0.50 | `step >= 15` |
| LLM explanation quality | 0.0–0.30 | Scored by LLM judge on `submit_diagnosis` |

---

## Difficulty Levels

| Level | Label | Bugs | Hints | Typical Steps |
|-------|-------|------|-------|---------------|
| 1 | Easy | 1 (obvious) | Full — names the suspicious column | 3–4 |
| 2 | Medium | 1 (subtle) | Partial — bug category mentioned | 5–8 |
| 3 | Hard | 2 (mixed) | Minimal — only bug count | 8–12 |
| 4 | Expert | 2 + adversarial column names | None | 10–14 |

---

## File Structure

```
ml-debugger-gym/
├── README.md
├── Dockerfile
├── requirements.txt
├── env.py                  # MLDebuggerEnv (OpenEnv BaseEnvironment)
├── actions.py              # 5 Pydantic Action models
├── observations.py         # Observation + ModelMetrics models
├── grader.py               # MLDebuggerRubric (programmatic + LLM scoring)
├── dataset_generator.py    # Procedural broken pipeline generation
├── test_agent.py           # Heuristic test agent (end-to-end sanity check)
├── demo.py                 # Demo script with argparse CLI
├── tests/
│   ├── test_env.py
│   └── test_grader.py
└── datasets/               # Pre-generated CSVs (optional)
```

---

## License

MIT
