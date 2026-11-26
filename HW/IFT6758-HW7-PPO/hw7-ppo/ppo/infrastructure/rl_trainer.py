import os
import time

import gymnasium as gym
import numpy as np
import torch

from ppo.agents.ppo_agent import PPOAgent
from ppo.infrastructure.utils import CSVLogger, explained_variance, set_seed


def make_env(env_name, seed, normalize_reward=False, gamma=0.99, render_mode=None):
    if env_name.startswith("MinAtar/"):
        from minatar.gym import register_envs
        if env_name not in gym.registry:
            register_envs()
    env = gym.make(env_name, render_mode=render_mode)
    if env_name.startswith("MinAtar/"):
        # MinAtar returns boolean images (10, 10, C); the network expects floats.
        env = gym.wrappers.TransformObservation(
            env, lambda o: o.astype(np.float32),
            gym.spaces.Box(0.0, 1.0, env.observation_space.shape, np.float32))
    # Records the true (un-normalised) episode return in info["episode"] when an episode ends.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if normalize_reward:
        # Scales rewards by a running estimate of the std of the discounted return.
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class RL_Trainer(object):
    """
    Collects on-policy rollouts from num_envs environments, computes advantages and
    updates the agent. This file is given to you; read it to understand the data your
    functions receive, but you do not need to modify it.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.alg = cfg.alg
        set_seed(cfg.logging.seed)
        device = "cuda" if (self.alg.use_gpu and torch.cuda.is_available()) else "cpu"

        n = cfg.env.num_envs
        self.envs = [make_env(cfg.env.env_name, cfg.logging.seed + i, cfg.env.normalize_reward, self.alg.gamma)
                     for i in range(n)]
        self.eval_env = make_env(cfg.env.env_name, cfg.logging.seed + 10_000)
        space = self.envs[0].action_space
        self.discrete = isinstance(space, gym.spaces.Discrete)
        self.ob_shape = self.envs[0].observation_space.shape
        ac_dim = int(space.n) if self.discrete else int(np.prod(space.shape))
        self.ac_shape = () if self.discrete else space.shape
        self.agent = PPOAgent(self.ob_shape, ac_dim, self.discrete, self.alg, device=device)
        self.logger = CSVLogger(cfg.logging.logdir)

    def _env_action(self, action):
        if self.discrete:
            return int(action)
        space = self.envs[0].action_space
        return np.clip(action, space.low, space.high)

    def collect_rollout(self, raw_obs):
        """Runs the current policy for num_steps steps in each environment."""
        T, N, agent, gamma = self.alg.num_steps, len(self.envs), self.agent, self.alg.gamma
        buf = {
            "obs": np.zeros((T, N) + self.ob_shape, dtype=np.float32),
            "actions": np.zeros((T, N) + self.ac_shape, dtype=np.float32),
            "log_probs": np.zeros((T, N), dtype=np.float32),
            "values": np.zeros((T, N), dtype=np.float32),
            "rewards": np.zeros((T, N), dtype=np.float32),
            "dones": np.zeros((T, N), dtype=np.float32),
        }
        episode_returns, episode_lengths = [], []
        for t in range(T):
            norm_obs = agent.normalize_obs(raw_obs, update=True)
            actions, log_probs = agent.get_action(norm_obs)
            buf["obs"][t], buf["actions"][t], buf["log_probs"][t] = norm_obs, actions, log_probs
            buf["values"][t] = agent.get_value(norm_obs)

            for i, env in enumerate(self.envs):
                ob, rew, terminated, truncated, info = env.step(self._env_action(actions[i]))
                if terminated or truncated:
                    episode_returns.append(float(info["episode"]["r"]))
                    episode_lengths.append(int(info["episode"]["l"]))
                    if truncated and not terminated:
                        # Time-limit truncation is not a real episode end: bootstrap with V(s_final).
                        rew = rew + gamma * agent.get_value(agent.normalize_obs(ob[None]))[0]
                    ob, _ = env.reset()
                buf["rewards"][t, i] = rew
                buf["dones"][t, i] = float(terminated or truncated)
                raw_obs[i] = ob

        last_value = agent.get_value(agent.normalize_obs(raw_obs))
        return buf, last_value, raw_obs, episode_returns, episode_lengths

    def evaluate(self, num_episodes, deterministic=None, render=False, max_steps=10_000):
        """Runs the policy without exploration noise on a separate, un-normalised environment."""
        env = make_env(self.cfg.env.env_name, self.cfg.logging.seed + 20_000, render_mode="rgb_array") \
            if render else self.eval_env
        if deterministic is None:
            deterministic = self.cfg.logging.eval_deterministic
        returns, frames = [], []
        for ep in range(num_episodes):
            ob, _ = env.reset()
            done, ep_ret, steps = False, 0.0, 0
            while not done and steps < max_steps:
                steps += 1
                if render and ep == 0:
                    frames.append(env.render())
                action, _ = self.agent.get_action(self.agent.normalize_obs(ob[None]), deterministic)
                ob, rew, terminated, truncated, _ = env.step(self._env_action(action[0]))
                ep_ret += rew
                done = terminated or truncated
            returns.append(ep_ret)
        return float(np.mean(returns)), frames

    def save_video(self, frames, name):
        import imageio
        path = os.path.join(self.cfg.logging.logdir, "videos")
        os.makedirs(path, exist_ok=True)
        fps = self.eval_env.metadata.get("render_fps") or 10
        frames = [np.asarray(f) for f in frames]
        if frames[0].dtype != np.uint8:  # MinAtar renders floats in [0, 1]
            frames = [(f * 255).astype(np.uint8) for f in frames]
        if frames[0].shape[0] < 100:  # upscale tiny MinAtar frames so the video is watchable
            k = 160 // frames[0].shape[0]
            frames = [np.kron(f, np.ones((k, k, 1), dtype=np.uint8)) for f in frames]
        imageio.mimsave(os.path.join(path, name + ".mp4"), frames, fps=fps, codec="libx264",
                        quality=7, macro_block_size=1)

    def run_training_loop(self):
        batch_size = self.alg.num_steps * len(self.envs)
        num_iterations = max(int(self.alg.total_timesteps) // batch_size, 1)
        raw_obs = np.stack([env.reset()[0] for env in self.envs]).astype(np.float32)
        start, env_steps, last_eval = time.time(), 0, float("nan")

        for it in range(1, num_iterations + 1):
            if self.alg.anneal_lr:
                self.agent.set_learning_rate_fraction(1.0 - (it - 1.0) / num_iterations)

            buf, last_value, raw_obs, ep_rets, ep_lens = self.collect_rollout(raw_obs)
            env_steps += batch_size

            advantages, value_targets = self.agent.estimate_advantages(
                buf["rewards"], buf["values"], buf["dones"], last_value)

            batch = {k: v.reshape((batch_size,) + v.shape[2:]) for k, v in buf.items()}
            batch["advantages"] = advantages.reshape(-1)
            if value_targets is not None:
                batch["value_targets"] = value_targets.reshape(-1)
            train_logs = self.agent.update(batch)

            eval_return = float("nan")
            if it % self.cfg.logging.eval_freq == 0 or it == num_iterations:
                eval_return, _ = self.evaluate(self.cfg.logging.eval_episodes)
                last_eval = eval_return

            row = {
                "iteration": it,
                "env_steps": env_steps,
                "train/episode_return": float(np.mean(ep_rets)) if ep_rets else float("nan"),
                "train/episode_length": float(np.mean(ep_lens)) if ep_lens else float("nan"),
                "eval/return": eval_return,
                "critic/explained_variance": explained_variance(batch["values"], batch["value_targets"])
                if value_targets is not None else float("nan"),
                "critic/value_loss": float("nan"),
                "time": time.time() - start,
                "sps": int(env_steps / (time.time() - start)),
            }
            row.update(train_logs)
            self.logger.log(row)
            print(f"[{self.cfg.env.exp_name}] it {it}/{num_iterations} steps {env_steps} "
                  f"train_return {row['train/episode_return']:.1f} eval_return {last_eval:.1f} "
                  f"expl_var {row['critic/explained_variance']:.3f} kl {row['policy/approx_kl']:.4f} "
                  f"sps {row['sps']}", flush=True)

        torch.save(self.agent.state_dict(), os.path.join(self.cfg.logging.logdir, "agent.pt"))
        if self.cfg.logging.video:
            _, frames = self.evaluate(1, render=True)
            self.save_video(frames, "final_policy")
        return self.logger.history()
