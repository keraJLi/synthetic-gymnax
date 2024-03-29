import pickle
from importlib import resources

# Import gymnax to overwrite gymnax.make, but use old version as gymnax_make
import gymnax
from gymnax import make as gymnax_make

from synthetic_gymnax.synthetic_environment import SynthEnv, SynthEnvParams

__all__ = ["SynthEnv", "synthetic_envs"]

CHECKPOINT_FOLDERS = {
    # Classic control
    "Acrobot-v1": "acrobot",
    "CartPole-v1": "cartpole",
    "ContinuousMountainCar-v0": "continuous_mountaincar",
    "MountainCar-v0": "mountaincar",
    "Pendulum-v0": "pendulum",
    # Brax
    "hopper": "hopper",
    "swimmer": "swimmer",
    "walker2d": "walker2d",
    "halfcheetah": "halfcheetah",
    "humanoidstandup": "humanoidstandup",
}


def load_checkpoint(eval_env_id):
    """Load checkpoint of SynthEnvParams from pickle file"""
    checkpoint_folder = CHECKPOINT_FOLDERS[eval_env_id]
    files = resources.files("synthetic_gymnax").joinpath("checkpoints")
    with files.joinpath(checkpoint_folder).joinpath("checkpoint.pkl").open("rb") as f:
        return pickle.load(f)


def make_synthenv_cls(eval_env_id):
    """Return new SynthEnv class where the default parameters are loaded from a checkpoint"""

    class CheckpointedSynthEnv(SynthEnv):
        @property
        def default_params(self):
            return SynthEnvParams(
                network_params=load_checkpoint(eval_env_id),
                max_steps_in_episode=self.max_steps_in_episode,
            )

    CheckpointedSynthEnv.__name__ = f"Synthetic{eval_env_id}"
    return CheckpointedSynthEnv


# synthetic_envs is a dict of {"Synthetic-<env_id>": SynthEnv}
synthetic_envs = {}

for env_id, folder in CHECKPOINT_FOLDERS.items():
    CheckpointedSynthEnv = make_synthenv_cls(env_id)
    synthetic_envs[f"Synthetic-{env_id}"] = CheckpointedSynthEnv


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's infamous env.make(env_name)"""
    if env_id.startswith("Synthetic-"):
        eval_env_id = env_id[len("Synthetic-") :]
        env = synthetic_envs[env_id](eval_env_id, **env_kwargs)
        return env, env.default_params

    return gymnax_make(env_id, **env_kwargs)


gymnax.make = make
