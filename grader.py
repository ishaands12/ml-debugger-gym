# grader.py
# MLDebuggerRubric — programmatic reward scoring.

import json


class MLDebuggerRubric:
    """
    Scores agent actions using programmatic rewards + optional LLM quality bonus.

    Reward schedule:
      -0.02  per step (always — reduced from 0.05 to allow more exploration)
      +0.05  first time each inspect target is used (exploration bonus)
      +0.05  first apply_fix attempt in episode (encourages trying fixes)
      +0.3   correct bug type on submit_diagnosis
      -0.2   wrong bug type on submit_diagnosis
      +0.5   target accuracy reached at submission
      delta*2  accuracy improvement on evaluate_model (positive delta only)
      -0.1   overfitting detected (train-val gap > 0.15)
      -0.5   timeout (applied externally by env)
      0–0.3  LLM quality score on submit_diagnosis (optional, requires OPENAI_API_KEY)
    """

    def __init__(self):
        self.llm_available = False
        self.llm_client = None
        try:
            import openai
            import os
            if os.environ.get("OPENAI_API_KEY"):
                self.llm_client = openai.OpenAI()
                self.llm_available = True
        except (ImportError, Exception):
            pass

        # Exploration tracking — reset per episode via reset_episode()
        self._seen_inspect_targets: set = set()
        self._fix_attempted: bool = False

    def reset_episode(self):
        """Call at the start of each episode to reset exploration trackers."""
        self._seen_inspect_targets = set()
        self._fix_attempted = False

    def _get_obs(self, state):
        """Return the Observation from either env wrapper or obs directly."""
        if hasattr(state, "_obs") and state._obs is not None:
            return state._obs
        if hasattr(state, "_state") and state._state is not None:
            return state._state
        return state

    def _ground_truth(self, state):
        return state._ground_truth_bug

    def programmatic_score(self, action, state) -> float:
        obs = self._get_obs(state)
        score = -0.02  # step penalty (reduced to allow exploration)

        if action.action_type == "inspect_data":
            # Dense bonus: reward first use of each inspection type
            key = f"{action.target}:{action.column or ''}"
            if key not in self._seen_inspect_targets:
                self._seen_inspect_targets.add(key)
                score += 0.05

        elif action.action_type == "apply_fix":
            # Dense bonus: reward first fix attempt in episode
            if not self._fix_attempted:
                self._fix_attempted = True
                score += 0.05

        elif action.action_type == "evaluate_model":
            delta = obs.current_metrics.accuracy - obs.baseline_metrics.accuracy
            score += max(0.0, delta * 2.0)
            gap = obs.current_metrics.train_accuracy - obs.current_metrics.val_accuracy
            if gap > 0.15:
                score -= 0.1

        elif action.action_type == "submit_diagnosis":
            if action.bug_type == self._ground_truth(state):
                score += 0.3
            else:
                score -= 0.2
            if obs.current_metrics.accuracy >= obs.target_accuracy:
                score += 0.5

        return round(score, 4)

    def llm_score(self, action, state) -> float:
        """Optional LLM quality bonus for submit_diagnosis explanations.
        Requires OPENAI_API_KEY environment variable. Gracefully returns 0.0 if unavailable.
        """
        if action.action_type != "submit_diagnosis" or not self.llm_available:
            return 0.0

        obs = self._get_obs(state)
        prompt = (
            f"You are evaluating an AI agent's diagnosis of a broken ML pipeline.\n\n"
            f"Bug explanation given by agent: \"{action.explanation}\"\n"
            f"Actual bug type: {self._ground_truth(state)}\n"
            f"Accuracy before fixes: {obs.baseline_metrics.accuracy:.3f}\n"
            f"Accuracy after fixes: {obs.current_metrics.accuracy:.3f}\n\n"
            f"Score on TWO dimensions:\n"
            f"1. clarity (0.0–0.15): Is the explanation specific and concrete?\n"
            f"2. reasoning (0.0–0.15): Does the reasoning logically lead to the diagnosis?\n\n"
            f'Return ONLY valid JSON: {{"clarity": <float>, "reasoning": <float>}}'
        )
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            scores = json.loads(response.choices[0].message.content)
            return round(
                float(scores.get("clarity", 0)) + float(scores.get("reasoning", 0)), 4
            )
        except Exception:
            return 0.0

    def forward(self, action, state) -> float:
        raw = self.programmatic_score(action, state) + self.llm_score(action, state)
        return max(0.01, min(0.99, raw))
