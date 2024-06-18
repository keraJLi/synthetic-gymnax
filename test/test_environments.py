import unittest

import gymnax
import jax

import synthetic_gymnax


class TestEnvironments(unittest.TestCase):
    def setUp(self):
        self.synth_envs = {
            env_id: gymnax.make(env_id)
            for env_id in synthetic_gymnax.synthetic_envs
        }

    def test_reset_step(self):
        for env_id, (env, params) in self.synth_envs.items():
            with self.subTest(env_id=env_id):
                self.assertIsInstance(env, synthetic_gymnax.SynthEnv)
                self.assertIsInstance(params, synthetic_gymnax.SynthEnvParams)

                rng = jax.random.PRNGKey(0)
                reset, step = jax.jit(env.reset), jax.jit(env.step)
                params = env.default_params

                try:
                    obs, state = reset(rng, params)
                except Exception as e:
                    self.fail(f"Failed to reset {env_id}: {e}")

                self.assertIsInstance(obs, jax.numpy.ndarray)

                action = env.action_space(params).sample(rng)
                try:
                    obs, new_state, reward, done, _ = step(rng, state, action, params)
                except Exception as e:
                    self.fail(f"Failed to step {env_id}: {e}")

                self.assertIsInstance(obs, jax.numpy.ndarray)

                self.assertIsInstance(reward, jax.numpy.ndarray)
                self.assertEqual(reward.shape, ())
                self.assertIsInstance(done.item(), bool)

                self.assertIsInstance(done, jax.numpy.ndarray)
                self.assertEqual(done.shape, ())
                self.assertIsInstance(reward.item(), float)

                def shapes_dtypes(x):
                    return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)

                state_sd = jax.tree_util.tree_map(shapes_dtypes, state)
                new_state_sd = jax.tree_util.tree_map(shapes_dtypes, new_state)
                self.assertEqual(state_sd, new_state_sd)
