import argparse
import os
import pickle
import time
from copy import deepcopy
from typing import Callable, Tuple

import chex
import jax
import numpy as np
import optax
import pandas as pd
import yaml
from dotmap import DotMap
from evosax import Strategies
from evosax.core.reshape import ravel_pytree
from jax import numpy as jnp
from purerl.algos import (
    DDPG,
    DQN,
    PPO,
    SAC,
    TD3,
    DDPGConfig,
    DQNConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
)
from purerl.evaluate import make_evaluate

from synthetic_gymnax import SynthEnv

LOG = []

RolloutFunction = Callable[[int, chex.ArrayTree, chex.PRNGKey], dict[str, chex.Scalar]]


def log(
    step_info: dict[str, chex.Numeric], log_fname: str, model: chex.ArrayTree = None
):
    """Rudimentary logging function, which logs to a csv and checkpoints the model."""
    LOG.append(step_info)

    print(f"Fitness of search distribution mean: {step_info['mean_fitness']:.2f}")

    path = os.path.join("logs", log_fname)
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(LOG).to_csv(os.path.join(path, "log.csv"), index=False)

    if model is not None:
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)


def get_agent(algo: str) -> Tuple[type, type]:
    """
    Returns the algorithm's class and config class, instead of pureRL's get_agent,
    which returns train_fn and the config class.
    """
    return {
        "ppo": (PPO, PPOConfig),
        "sac": (SAC, SACConfig),
        "dqn": (DQN, DQNConfig),
        "ddpg": (DDPG, DDPGConfig),
        "td3": (TD3, TD3Config),
    }[algo]


def make_sample_config(
    config_cls, config_dict, synth_env, synth_env_params
) -> Callable[[chex.PRNGKey], chex.ArrayTree]:
    """
    Returns a function that samples a random hyperparameter configuration.
    HPs are sampled uniformly from lists which are given in config_dict.
    """

    def is_leaf(val):
        return not isinstance(val, dict)

    def sample(val, rng):
        if isinstance(val, list):
            return jax.random.choice(rng, jax.numpy.array(val))
        else:
            return val

    def sample_train_config(rng):
        """Takes a dict of hyperparameters for the inner loop algorithm. If any value
        is a list, it samples a random one from the list."""
        leaves, treedef = jax.tree_util.tree_flatten(config_dict, is_leaf=is_leaf)
        rngs = list(jax.random.split(rng, len(leaves)))
        leaves = [sample(leaf, rng) for leaf, rng in zip(leaves, rngs)]
        sampled_dict = jax.tree_util.tree_unflatten(treedef, leaves)
        config = config_cls.from_dict(sampled_dict)
        config = config.replace(
            env=synth_env,
            env_params=synth_env_params,
            skip_initial_evaluation=True,
        )
        return config

    return sample_train_config


def make_rollout(
    algo_cls: type,
    sample_train_config: Callable[[chex.PRNGKey], type],
    eval_env,
    eval_env_params,
    eval_config,
) -> RolloutFunction:
    # The array resulting from a call to rollout will have shape
    # (num_synth_envs, num_inits)

    eval_config = deepcopy(eval_config)

    num_eval_envs = eval_config.pop("num_eval_envs")
    schedule_fn_str = eval_config.pop("schedule_fn", "linear_schedule")
    schedule_fn = getattr(optax, schedule_fn_str)
    eval_step_schedule = schedule_fn(**eval_config)

    def rollout(gen, meta_params, rng):
        """Returns shape (devices, synth_envs, rollouts).squeeze(0)"""
        rng_config, rng_train = jax.random.split(rng)

        # Calculate number of evaluation steps in this generation
        num_eval_steps = eval_step_schedule(gen).astype(int)
        params = eval_env_params.replace(max_steps_in_episode=num_eval_steps)
        evaluate = make_evaluate(eval_env, params, num_eval_envs, num_eval_steps)

        # Sample train config hyperparameters
        train_config = sample_train_config(rng_config)
        params = train_config.env_params.replace(network_params=meta_params)

        # Roll out
        config = train_config.replace(env_params=params, eval_callback=evaluate)
        _, (lengths, returns) = algo_cls.train(config, rng=rng_train)

        # lengths, returns have shape (eval_step, eval_envs)
        metrics = {
            "final_length": lengths.mean(axis=1)[-1],
            "final_return": returns.mean(axis=1)[-1],
        }
        return metrics

    return rollout


def get_rollouts(
    config: DotMap, synth_env, synth_env_params
) -> dict[str, Tuple[RolloutFunction, RolloutFunction]]:
    """ 
    Generates and vmaps/pmaps a rollout function for population and mean of the search
    distributions, and for each algorithm.
    """
    eval_env, eval_env_params = synth_env.eval_env, synth_env.eval_env_params
    rollouts = {}
    algos = config.inner_loop_algorithm
    algos = [algos] if isinstance(config.inner_loop_algorithm, str) else algos
    for algo in algos:
        algo_cls, config_cls = get_agent(algo)
        rl_config = config.inner_config[algo]
        sample_config = make_sample_config(
            config_cls, deepcopy(rl_config.toDict()), synth_env, synth_env_params
        )
        rollout = make_rollout(
            algo_cls,
            sample_config,
            eval_env,
            eval_env_params,
            config.eval_config,
        )

        # Vmap across synth envs and rollouts, pmap across synth envs (if #devices > 1)
        rollout_pop = jax.vmap(rollout, in_axes=(None, None, 0))
        rollout_pop = jax.vmap(rollout_pop, in_axes=(None, 0, None))
        if jax.device_count() > 1:
            rollout_pop = jax.pmap(rollout_pop, in_axes=(None, 0, None))
        else:
            rollout_pop = jax.jit(rollout_pop)

        # Vmap across rollouts, pmap across synth envs and rollouts (synth envs are
        # stacked, such that each device gets the population mean)
        rollout_mean = jax.vmap(rollout, in_axes=(None, None, 0))
        if jax.device_count() > 1:
            rollout_mean = jax.pmap(rollout_mean, in_axes=(None, 0, 0))
        else:
            rollout_mean = jax.jit(rollout_mean)

        rollouts[algo] = rollout_pop, rollout_mean

    return rollouts


def train(config: DotMap):
    """Run meta-evolution of synthetic environment."""
    synth_env_kwargs = config.pop("synth_env_kwargs", None) or {}
    synth_env = SynthEnv(config.env_name, **synth_env_kwargs)
    synth_env_params = synth_env.default_params

    rollouts = get_rollouts(config, synth_env, synth_env_params)

    def get_algos(gen):
        return {
            a: rollouts[a]
            for i, a in enumerate(rollouts.keys())
            if gen % len(rollouts) == i
        }

    # Setup meta-es strategy & init its state
    rng = jax.random.PRNGKey(config.seed_id)
    rng, rng_init = jax.random.split(rng)
    strategy = Strategies[config.strategy_name](
        pholder_params=synth_env_params.network_params,
        maximize=True,
        **config.strategy_params,
    )
    state = strategy.initialize(rng_init)

    # Replace mean of strategy by NN initialization (default params of synthenv)
    state = state.replace(mean=ravel_pytree(synth_env_params.network_params))
    print("Initialized ES strategy")

    # Run outer loop meta-es over inner loop synth env PPO training runs
    for gen in range(config.meta_config.num_generations):
        start_time = time.time()

        # 1. SAMPLE POPULATION: Get new population of meta synth env parameters from ES
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        meta_params, state = strategy.ask(rng_ask, state)

        # 2. EXECUTE INNER LOOP: Eval synth env params on multi inner loop PPO trainings
        info, fitness = {}, []
        batch_rng = jax.random.split(rng_eval, config.meta_config.num_rollouts)
        for algo, (rollout_pop, _) in get_algos(gen).items():
            print(f"Gen {gen} - Rolling out {algo}")
            info_algo = rollout_pop(gen, meta_params, batch_rng)
            fitness.append(info_algo["final_return"])

            for k, v in info_algo.items():
                info[f"{k}/{algo}"] = v.mean()

        # Compute fitness by averaging over algorithms and seeds
        axis = (0, 3) if jax.device_count() > 1 else (0, 2)
        fitness = np.mean(fitness, axis=axis).reshape(-1)

        # 3. UPDATE META PARAMS: Update the ES with the fitness (perf of current algo)
        state = strategy.tell(meta_params, fitness, state)
        fitness = jax.block_until_ready(fitness)

        step_info = {
            "num_gens": gen + 1,
            "time_gen": time.time() - start_time,
            "best_fitness": state.best_fitness,
            "population_fitness": fitness.mean(),
            "population_fitness_std": fitness.std(),
            **jax.tree_map(lambda x: x.mean(), info),
        }

        # 4. EVALUATE POPULATION MEAN
        model = strategy.get_eval_params(state)

        # Calculate fitness of strategy mean every mean_eval_freq steps
        if (
            gen % config.meta_config.mean_eval_freq == 0
            or gen + 1 == config.meta_config.num_generations
        ):
            rng, rng_eval = jax.random.split(rng, 2)
            rng_eval = jax.random.split(rng_eval, config.meta_config.num_rollouts_mean)
            if jax.device_count() > 1:
                rng_eval = rng_eval.reshape(jax.device_count(), -1, 2)
                stacked_model = jax.tree_map(
                    lambda x: jnp.stack([x] * jax.device_count()),
                    model,
                )
            else:
                stacked_model = model

            info, fitness = {}, []
            for algo, (_, rollout_mean) in rollouts.items():
                info_algo = rollout_mean(gen, stacked_model, rng_eval)
                fitness.append(info_algo["final_return"])
                for k, v in info_algo.items():
                    info[f"{k}/{algo}"] = v.mean()

            step_info = {
                **step_info,
                "mean_fitness": np.mean(fitness),
                "mean_fitness_std": np.std(fitness),
                **{f"mean_{k}": v for k, v in info.items()},
            }

        # 5. LOGGING
        log(step_info, config.log_fname, model=model)


parser = argparse.ArgumentParser(description="Meta-evolve a synthetic environment.")
parser.add_argument("--config", type=str, default="acrobot.yaml")


def main():
    """
    Entry point of the script. Executes train with a yaml config given as a command line
    argument. For example, the script runs with examples/acrobot.yaml and
    examples/hopper.yaml.
    """
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = DotMap(yaml.safe_load(f))

    env = config.env_name.replace("/", "_")
    if isinstance(config.inner_loop_algorithm, str):
        ila = config.inner_loop_algorithm
    else:
        ila = "_".join(config.inner_loop_algorithm)
    log_fname = f"{env}_{ila}"

    config.log_fname = log_fname
    train(config)


if __name__ == "__main__":
    main()
