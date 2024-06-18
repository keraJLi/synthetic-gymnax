from typing import Any, Dict, Tuple

import chex
import distrax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import _envs as brax_envs
from flax import struct
from flax.core.frozen_dict import FrozenDict
from gymnax import registered_envs as gymnax_envs
from gymnax.environments import spaces
from gymnax.environments.environment import Environment
from rejax.brax2gymnax import Brax2GymnaxEnv

from synthetic_gymnax.synthenv_network import SynthEnvMLP


def space_dim(space):
    if isinstance(space, spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, spaces.Discrete):
        return space.n
    else:
        raise NotImplementedError("Unsupported space")


@struct.dataclass
class SynthEnvParams:
    network_params: FrozenDict
    max_steps_in_episode: int


@struct.dataclass
class SynthEnvState:
    obs: chex.Array
    done: bool
    time: int = 0


class SynthEnv(Environment):
    """Uses SynthEnvMLP which takes in obs (or vectorized state) and action"""

    def __init__(
        self,
        eval_env: str,
        max_steps_in_episode: int = 1,
        network_kwargs: Dict[str, Any] = None,
        eval_env_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        network_kwargs = network_kwargs or {}
        eval_env_kwargs = eval_env_kwargs or {}

        if eval_env in gymnax_envs:
            self.eval_env, self.eval_env_params = gymnax.make(
                eval_env, **eval_env_kwargs
            )
        elif eval_env in brax_envs:
            self.eval_env = Brax2GymnaxEnv(eval_env, **eval_env_kwargs)
            self.eval_env_params = self.eval_env.default_params
        else:
            raise ValueError(f"Unknown eval_env: {eval_env}")

        self.max_steps_in_episode = max_steps_in_episode

        if "features" in network_kwargs.keys():
            # Convert to tuple to make immutable
            network_kwargs["features"] = tuple(network_kwargs["features"])

        # Set up latent distribution
        obs_size = space_dim(self.observation_space(self.eval_env_params))
        latent_size = network_kwargs.pop("latent_size", (obs_size,))
        if isinstance(latent_size, int):
            latent_size = (latent_size,)

        latent_dist = network_kwargs.pop("latent_dist", "normal")
        if latent_dist == "normal":
            latent_dist = distrax.MultivariateNormalDiag(
                loc=jnp.zeros(latent_size), scale_diag=jnp.ones(latent_size)
            )
        elif latent_dist == "categorical":
            latent_dist = distrax.OneHotCategorical(logits=jnp.zeros(latent_size))
        elif latent_dist == "uniform":
            latent_dist = distrax.Uniform(
                low=jnp.zeros(latent_size), high=jnp.ones(latent_size)
            )
        elif latent_dist == "softmax":
            probs = jax.nn.softmax(-jnp.arange(latent_size[0]))
            latent_dist = distrax.OneHotCategorical(probs=probs)
        else:
            raise ValueError("Unknown latent_dist")
        network_kwargs["latent_dist"] = latent_dist

        # Set network
        self.network = SynthEnvMLP(obs_size, **network_kwargs)

    @property
    def default_params(self) -> SynthEnvParams:
        params = self.network.init(
            jax.random.PRNGKey(0),
            jax.random.PRNGKey(0),
            jnp.empty(space_dim(self.eval_env.observation_space(self.eval_env_params))),
            jnp.empty(space_dim(self.eval_env.action_space(self.eval_env_params))),
            only_reward=self.max_steps_in_episode == 1,
        )
        return SynthEnvParams(params, max_steps_in_episode=self.max_steps_in_episode)

    def step_env(
        self, key: chex.PRNGKey, state: SynthEnvState, action, params: SynthEnvParams
    ) -> Tuple[chex.Array, SynthEnvState, chex.Scalar, bool]:
        if isinstance(self.action_space(params), spaces.Discrete):
            action = jax.nn.one_hot(action, self.action_space(params).n)

        only_reward = self.max_steps_in_episode == 1
        transition = self.network.apply(
            params.network_params,
            state.obs,
            action,
            only_reward=only_reward,
            method="step",
        )
        transition = jax.tree_map(lambda x: x.squeeze(0), transition)
        time = state.time + 1
        done = time >= self.max_steps_in_episode

        if only_reward:
            reward = transition
            next_obs = jnp.empty_like(state.obs)  # Don't care due to autoreset
        else:
            reward, next_obs, pred_done = transition
            done = jnp.logical_or(done, pred_done > 0)

        next_obs = next_obs.reshape(
            self.eval_env.observation_space(self.eval_env_params).shape
        )
        state = state.replace(time=time, obs=next_obs, done=done)
        return (
            next_obs,
            state,
            reward,
            self.is_terminal(state, params),
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: SynthEnvParams
    ) -> Tuple[chex.Array, SynthEnvState]:
        initial_obs = self.network.apply(params.network_params, key, method="reset")
        return initial_obs, SynthEnvState(
            time=0,
            obs=initial_obs,
            done=False,
        )

    def get_obs(self, state: SynthEnvState) -> chex.Array:
        return state.obs

    def is_terminal(self, state: SynthEnvState, params: SynthEnvParams) -> bool:
        return state.done

    @property
    def name(self) -> str:
        return "ObsSynthEnv-v0"

    @property
    def num_actions(self) -> int:
        return space_dim(self.action_space)

    def action_space(self, params: SynthEnvParams) -> spaces.Space:
        return self.eval_env.action_space(self.eval_env_params)

    def observation_space(self, params: SynthEnvParams) -> spaces.Space:
        return self.eval_env.observation_space(self.eval_env_params)

    def state_space(self, params: SynthEnvParams) -> spaces.Space:
        return spaces.Box(-jnp.inf, jnp.inf, (space_dim(self.observation_space),))
