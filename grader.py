# grader.py
# MLDebuggerRubric — hybrid programmatic + LLM reward scoring.
# Kept separate from env.py for testability.

import json


class MLDebuggerRubric:
    """
    Scores agent actions using both programmatic checks and LLM evaluation.

    Reward schedule:
      -0.05  per step (always)
      +0.3   correct bug type on submit_diagnosis
      -0.2   wrong bug type on submit_diagnosis
      +0.5   target accuracy reached at submission
      delta*2  accuracy improvement on evaluate_model
      -0.1   overfitting detected (train-val gap > 0.15)
      -0.5   timeout (applied externally by env)
      0–0.3  LLM quality score on submit_diagnosis explanation
    """

    def __init__(self):
        self.llm_available = False
        self.llm_client = None
        try:
            import openai
            self.llm_client = openai.OpenAI()
            self.llm_available = True
        except (ImportError, Exception):
            pass

    def _get_obs(self, state):
        """Return the Observation, whether state is the env or the obs directly."""
        if hasattr(state, "_obs") and state._obs is not None:
            return state._obs
        if hasattr(state, "_state") and state._state is not None:
            return state._state
        return state

    def _ground_truth(self, state):
        """Return the ground truth bug label from either env or a FakeState."""
        return state._ground_truth_bug

    def programmatic_score(self, action, state) -> float:
        obs = self._get_obs(state)
        score = -0.05  # step penalty

        if action.action_type == "evaluate_model":
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
        if action.action_type != "submit_diagnosis" or not self.llm_available:
            return 0.0

        obs = self._get_obs(state)
        prompt = (
            f"You are evaluating an AI agent's diagnosis of a broken ML pipeline.\n\n"
            f"Bug explanation given by agent: \"{action.explanation}\"\n"
            f"Actual bug type: {self._ground_truth(state)}\n"
            f"Accuracy before fixes: {obs.baseline_metrics.accuracy:.3f}\n"
            f"Accuracy after fixes: {obs.current_metrics.accuracy:.3f}\n\n"
            f"Score the explanation on TWO dimensions:\n"
            f"1. clarity (0.0 to 0.15): Is the explanation specific, concrete, and understandable?\n"
            f"2. reasoning (0.0 to 0.15): Does the explanation show logical reasoning from observations to the diagnosis?\n\n"
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
            return float(scores.get("clarity", 0)) + float(scores.get("reasoning", 0))
        except Exception:
            return 0.0

    def forward(self, action, state) -> float:
        return self.programmatic_score(action, state) + self.llm_score(action, state)
