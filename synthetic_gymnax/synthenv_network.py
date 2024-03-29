from collections.abc import Sequence
from typing import Callable

import chex
import distrax
from flax import linen as nn
from jax import numpy as jnp


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable[[chex.Array], chex.Array]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class SynthEnvMLP(nn.Module):
    state_size: int
    latent_dist: distrax.Distribution
    features: Sequence[int] = (32,)
    activation: Callable = nn.relu

    def setup(self):
        self.initial_state = MLP([*self.features, self.state_size], self.activation)
        self.reward = MLP([*self.features, 1], self.activation)
        self.next_state_delta = MLP([*self.features, self.state_size], self.activation)
        self.done = MLP([*self.features, 1], self.activation)

    def __call__(self, rng, state, action, only_reward=False):
        # If only_reward, parameters of self.next_state_delta and self.done are not used
        # such that they are excluded from meta-optimization
        return *self.reset(rng), *self.step(state, action, only_reward=only_reward)

    def reset(self, rng):
        z = self.latent_dist.sample(seed=rng)
        x = self.initial_state(z)
        return x

    def step(self, state, action, only_reward=False):
        x = jnp.hstack([state, action])
        reward = self.reward(x)
        if only_reward:
            return reward
        next_state = state + self.next_state_delta(x)
        done = self.done(next_state)
        return reward, next_state, done
