"""
FastAPI application for the ML Debugger Gym environment.
Serves the environment via HTTP and WebSocket endpoints compatible with OpenEnv clients.
"""

import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


app = create_app(
    MLDebuggerEnv,
    ApplyFixAction,   # representative action class for schema; env handles all types internally
    Observation,
    env_name="ml_debugger_gym",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
