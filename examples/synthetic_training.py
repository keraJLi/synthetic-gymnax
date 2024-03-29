from time import time

import jax
from purerl.algos import get_agent
from purerl.evaluate import make_evaluate

# fmt: off
# Importing synthetic_gymnax overwrites gymnax.make, which allows us to make
# Synthetic-CartPole-v1
import synthetic_gymnax

# fmt: on

# Load some config (set arbitrarily, synthetic environment should be pretty general)
train_fn, config_cls = get_agent("ppo")
config = config_cls.from_dict(
    {
        # Remove the "Synthetic-" prefix to train in the evaluation environment
        # directly. This should be slightly slower and lead to a low return.
        "env": "Synthetic-CartPole-v1",
        "agent_kwargs": {
            "activation": "tanh",
        },
        "num_envs": 100,
        "num_steps": 100,
        "num_epochs": 5,
        "num_minibatches": 10,
        "learning_rate": 0.005,
        "max_grad_norm": 10,
        "total_timesteps": 5000,
        "eval_freq": 1000,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    }
)


# Train the agent
start = time()
rng = jax.random.PRNGKey(0)
train_state, _ = jax.jit(train_fn)(config, rng)
end = time()

if isinstance(config.env, synthetic_gymnax.SynthEnv):
    # Evaluation environment is an attribute of the synthetic environment
    eval_env, eval_env_params = config.env.eval_env, config.env.eval_env_params
else:
    eval_env, eval_env_params = config.env, config.env_params

# Evaluate the trained agent
_, returns = make_evaluate(eval_env, eval_env_params, 200)(config, train_state, rng)

# Print some info
print("Finished training.")
print(f"Wall clock time: {end - start:.1f}s")
print(
    "Return on 200 seeds in the evaluation environment: "
    f"{returns.mean():g}±{returns.std():g}std"
)
