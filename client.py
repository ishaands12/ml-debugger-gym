"""ML Debugger Gym Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from actions import Action, InspectDataAction
from observations import Observation


class MLDebuggerEnvClient(EnvClient[Action, Observation, State]):
    """
    Client for the ML Debugger Gym environment.

    Example:
        >>> with MLDebuggerEnvClient(base_url="http://localhost:8000") as client:
        ...     obs = client.reset(difficulty=1)
        ...     obs = client.step(InspectDataAction(action_type="inspect_data", target="describe"))
    """

    def _step_payload(self, action: Action) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        obs_data = payload.get("observation", payload)
        observation = Observation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
