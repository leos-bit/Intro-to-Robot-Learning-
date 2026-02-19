from gymnasium.envs.registration import register, registry
from drone_training.src.envs.drone_env import Drone_Env
ENV_ID = "DroneEnv-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point=Drone_Env,
    )

__all__ = ["Drone_Env", "ENV_ID"]


