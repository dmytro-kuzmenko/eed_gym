"""EED Gym: Empathic Ethical Disobedience benchmark."""

from gymnasium.envs.registration import register, registry

__version__ = "1.1.0"

ENV_ID = "EEDGym-v1"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="eed_benchmark.envs.empathic_disobedience_env:EmpathicDisobedienceEnv",
    )

__all__ = ["ENV_ID", "__version__"]
