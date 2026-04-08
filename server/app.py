"""
FastAPI application for the ML Debugger Gym environment.
Serves the environment via HTTP and WebSocket endpoints compatible with OpenEnv clients.
"""

import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pydantic import TypeAdapter
from openenv.core.env_server.http_server import create_app

from actions import (
    Action,
    InspectDataAction,
    RunCodeAction,
    ApplyFixAction,
    EvaluateModelAction,
    SubmitDiagnosisAction,
)
from observations import Observation
from env import MLDebuggerEnv


# Action is a type alias (Annotated[Union[...]]), not a class.
# openenv-core's deserialize_action() calls action_cls.model_validate(),
# so we wrap it in a class that delegates to a TypeAdapter.
_action_ta = TypeAdapter(Action)


class ActionModel:
    """Bridge between the discriminated-union Action type and openenv-core's
    expected interface (must have .model_validate classmethod)."""

    @classmethod
    def model_validate(cls, data, **kwargs):
        return _action_ta.validate_python(data)

    @classmethod
    def model_json_schema(cls, **kwargs):
        return _action_ta.json_schema(**kwargs)


app = create_app(
    MLDebuggerEnv,
    ActionModel,
    Observation,
    env_name="ml_debugger_gym",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
