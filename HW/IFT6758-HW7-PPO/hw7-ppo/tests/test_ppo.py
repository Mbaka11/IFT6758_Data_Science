import os
import tempfile
import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir

try:
    from gradescope_utils.autograder_utils.decorators import weight
except ImportError:  # running locally without gradescope-utils
    def weight(_):
        return lambda f: f

from ppo.agents.ppo_agent import PPOAgent, compute_advantages, compute_discounted_returns, pg_loss, ppo_clip_loss
from ppo.critics.value_critic import ValueCritic, value_loss

CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")

# A small rollout with T = 5 steps and N = 2 environments.
REWARDS = np.array([[1.0, 0.5], [0.0, -1.0], [2.0, 0.0], [1.0, 1.0], [-0.5, 2.0]], dtype=np.float32)
DONES = np.array([[0, 0], [0, 1], [1, 0], [0, 0], [0, 1]], dtype=np.float32)
VALUES = np.array([[0.5, 0.2], [1.0, -0.3], [0.8, 0.1], [0.0, 0.4], [0.3, 0.9]], dtype=np.float32)
LAST_VALUE = np.array([1.5, -2.0], dtype=np.float32)


def load_cfg(*overrides):
    """Loads conf/config.yaml with Hydra, e.g. load_cfg("env=cartpole", "alg.rl_alg=pg")."""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        return compose(config_name="config", overrides=list(overrides))


class TestReturnsAndAdvantages(unittest.TestCase):

    @weight(5)
    def test_discounted_returns_single_episode(self):
        rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        out = compute_discounted_returns(rewards, np.zeros_like(rewards), np.zeros(1, dtype=np.float32), 0.5)
        self.assertEqual(out.shape, (3, 1))
        np.testing.assert_allclose(out[:, 0], [1.75, 1.5, 1.0], atol=1e-5)

    @weight(10)
    def test_discounted_returns_dones_and_bootstrap(self):
        out = compute_discounted_returns(REWARDS, DONES, LAST_VALUE, 0.9)
        expected = [[2.62, -0.4], [1.8, -1.0], [2.0, 2.52], [1.765, 2.8], [0.85, 2.0]]
        np.testing.assert_allclose(out, expected, atol=1e-4)

    @weight(10)
    def test_compute_advantages(self):
        returns = compute_discounted_returns(REWARDS, DONES, LAST_VALUE, 0.9)
        adv, targets = compute_advantages(returns, VALUES)
        self.assertEqual(adv.shape, (5, 2))
        mc = np.array([[2.62, -0.4], [1.8, -1.0], [2.0, 2.52], [1.765, 2.8], [0.85, 2.0]])
        np.testing.assert_allclose(adv, mc - VALUES, atol=1e-4)
        np.testing.assert_allclose(targets, mc, atol=1e-4)

    @weight(10)
    def test_estimate_advantages(self):
        # With a baseline: bootstrapped returns minus V(s_t). Without: reward-to-go, no bootstrap.
        for use_baseline in [True, False]:
            cfg = load_cfg("env=cartpole", f"alg.use_baseline={use_baseline}", "alg.gamma=0.9")
            agent = PPOAgent(4, 2, True, cfg.alg)
            adv, targets = agent.estimate_advantages(REWARDS, VALUES, DONES, LAST_VALUE)
            if use_baseline:
                mc = np.array([[2.62, -0.4], [1.8, -1.0], [2.0, 2.52], [1.765, 2.8], [0.85, 2.0]])
                np.testing.assert_allclose(adv, mc - VALUES, atol=1e-4)
                np.testing.assert_allclose(targets, mc, atol=1e-4)
            else:
                expected = [[2.62, -0.4], [1.8, -1.0], [2.0, 2.52], [0.55, 2.8], [-0.5, 2.0]]
                np.testing.assert_allclose(adv, expected, atol=1e-4)
                self.assertIsNone(targets)


class TestLosses(unittest.TestCase):

    @weight(5)
    def test_pg_loss(self):
        log_probs = torch.tensor([-1.0, -0.5, -2.0, -0.1])
        adv = torch.tensor([1.0, -2.0, 0.5, 3.0])
        self.assertAlmostEqual(pg_loss(log_probs, adv).item(), 0.325, places=4)

    @weight(10)
    def test_ppo_clip_loss(self):
        new = torch.tensor([-1.0, -0.5, -2.0, -0.1])
        old = torch.tensor([-1.2, -1.0, -1.5, -0.1])
        adv = torch.tensor([1.0, -2.0, 0.5, 3.0])
        loss, ratio = ppo_clip_loss(new, old, adv, 0.2)
        np.testing.assert_allclose(ratio.numpy(), np.exp([0.2, 0.5, -0.5, 0.0]), atol=1e-5)
        self.assertAlmostEqual(loss.item(), -0.3014557, places=4)

    @weight(5)
    def test_ppo_clip_gradient_is_zero_when_clipped(self):
        # ratio = e^0.5 > 1 + eps with a positive advantage: the clipped term is active,
        # so the loss should not push the log-probability any further.
        new = torch.tensor([-0.5], requires_grad=True)
        loss, _ = ppo_clip_loss(new, torch.tensor([-1.0]), torch.tensor([1.0]), 0.2)
        loss.backward()
        self.assertAlmostEqual(new.grad.item(), 0.0, places=6)

    @weight(5)
    def test_value_loss(self):
        new = torch.tensor([1.0, 2.0, -1.0])
        targets = torch.tensor([1.5, 0.0, 0.0])
        old = torch.tensor([0.0, 2.5, -1.1])
        self.assertAlmostEqual(value_loss(new, targets).item(), 0.875, places=4)
        self.assertAlmostEqual(value_loss(new, targets, old, 0.2).item(), 1.33, places=4)


class TestTraining(unittest.TestCase):

    @weight(10)
    def test_critic_fits_targets(self):
        torch.manual_seed(0)
        critic = ValueCritic(ob_shape=3, learning_rate=1e-2, update_epochs=50, num_minibatches=4)
        obs = torch.randn(256, 3)
        targets = obs @ torch.tensor([1.0, -2.0, 0.5]) + 1.0
        before = value_loss(critic(obs), targets).item()
        critic.update(obs, targets, critic(obs).detach())
        after = value_loss(critic(obs), targets).item()
        self.assertLess(after, 0.1 * before)

    @weight(5)
    def test_agent_update_changes_policy(self):
        for rl_alg in ["pg", "ppo"]:
            torch.manual_seed(0)
            cfg = load_cfg("env=cartpole", f"alg.rl_alg={rl_alg}")
            agent = PPOAgent(4, 2, True, cfg.alg)
            before = [p.detach().clone() for p in agent.policy.parameters()]
            obs = np.random.randn(64, 4).astype(np.float32)
            actions, log_probs = agent.get_action(obs)
            batch = {"obs": obs, "actions": actions, "log_probs": log_probs,
                     "values": agent.get_value(obs), "advantages": np.random.randn(64).astype(np.float32),
                     "value_targets": np.random.randn(64).astype(np.float32)}
            logs = agent.update(batch)
            self.assertIn("policy/loss", logs)
            changed = any(not torch.allclose(a, b) for a, b in zip(before, agent.policy.parameters()))
            self.assertTrue(changed, f"{rl_alg}: the policy parameters did not change")

    @weight(10)
    def test_ppo_learns_cartpole(self):
        from ppo.infrastructure.rl_trainer import RL_Trainer
        with tempfile.TemporaryDirectory() as logdir:
            cfg = load_cfg("env=cartpole", "env.exp_name=test", "alg.total_timesteps=30000",
                           f"logging.logdir={logdir}", "logging.eval_freq=1000")
            history = RL_Trainer(cfg).run_training_loop()
        self.assertGreaterEqual(history["eval/return"][-1], 195.0)


if __name__ == "__main__":
    unittest.main()
