import numpy as np
import torch
from torch import nn

from ppo.critics.value_critic import ValueCritic
from ppo.infrastructure.utils import RunningMeanStd
from ppo.policies.MLP_policy import MLPPolicy

#####################################################################
# Data layout
#
# A rollout contains T = num_steps steps from each of N = num_envs parallel
# environments. Every array below has shape (T, N), with time first.
#
#   rewards[t, i]: reward received after taking action t in environment i.
#   dones[t, i]:   1.0 if the episode in environment i ENDED at step t (the next
#                  observation belongs to a new episode), else 0.0.
#   values[t, i]:  V(s_t) predicted by the critic for the observation at step t.
#   last_value[i]: V(s_T), the value of the observation AFTER the last step of the
#                  rollout. It is used to bootstrap episodes that are still running
#                  when the rollout ends. Shape (N,).
#
# The rollout code has already handled time-limit truncation for you: when an
# episode is cut off by a time limit, gamma * V(final observation) is added to
# the reward of the last step and the step is marked done. So you can treat
# every done as a true episode end.
#####################################################################


def compute_discounted_returns(rewards, dones, last_value, gamma):
    """
    Discounted reward-to-go, computed backwards in time:

        R_t = r_t + gamma * (1 - done_t) * R_{t+1},   with R_T = last_value.

    Arguments: numpy arrays rewards (T, N), dones (T, N), last_value (N,); float gamma.
    Returns:   numpy array of returns, shape (T, N).
    """
    # TODO: implement the backwards recursion above.
    raise NotImplementedError


def compute_advantages(returns, values):
    """
    Advantage estimate with a learned baseline:

        A_t = R_t - V(s_t)

    and the regression targets for the value function, V_target(s_t) = R_t.

    Arguments: numpy arrays returns (T, N) (from compute_discounted_returns, bootstrapped with
               last_value) and values (T, N).
    Returns:   (advantages, value_targets), both numpy arrays of shape (T, N).
    """
    # TODO
    raise NotImplementedError


def pg_loss(log_probs, advantages):
    """
    Vanilla policy-gradient (REINFORCE) surrogate loss, whose gradient is
        -E[ grad log pi(a_t|s_t) * A_t ].

    Arguments: torch tensors log_probs (batch,), advantages (batch,).
    Returns:   a scalar tensor (the loss to MINIMISE).
    """
    # TODO
    raise NotImplementedError


def ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_coef):
    """
    PPO clipped surrogate loss (Schulman et al. 2017, eq. 7):

        ratio_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
        L       = -mean( min( ratio_t * A_t, clip(ratio_t, 1 - eps, 1 + eps) * A_t ) )

    Arguments: torch tensors new_log_probs (batch,), old_log_probs (batch,),
               advantages (batch,); float clip_coef (eps).
    Returns:   (loss, ratio). loss is a scalar tensor to MINIMISE; ratio has shape (batch,).
    HINT: compute the ratio from log-probabilities with exp(new - old) for numerical stability.
    """
    # TODO
    raise NotImplementedError


class PPOAgent(object):
    """
    On-policy actor-critic agent. With alg.rl_alg=ppo it trains the policy with the clipped
    PPO objective; with alg.rl_alg=pg it takes one vanilla policy-gradient step per batch.
    With alg.use_baseline=true a learned value function V(s) is used as the baseline.
    """

    def __init__(self, ob_shape, ac_dim, discrete, alg, device="cpu"):
        self.alg = alg
        self.device = torch.device(device)
        self.discrete = discrete
        net = alg.network
        self.policy = MLPPolicy(ob_shape, ac_dim, discrete, tuple(net.layer_sizes), net.activation,
                                alg.init_log_std).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=alg.learning_rate, eps=1e-5)
        self.critic = None
        if alg.use_baseline:
            self.critic = ValueCritic(
                ob_shape, tuple(net.layer_sizes), net.activation,
                learning_rate=alg.critic_learning_rate,
                update_epochs=alg.critic_update_epochs,
                num_minibatches=alg.critic_num_minibatches,
                max_grad_norm=alg.max_grad_norm,
                clip_coef=alg.clip_coef if alg.clip_vloss else None,
            ).to(self.device)
        self.obs_rms = RunningMeanStd(shape=tuple(np.atleast_1d(ob_shape))) if alg.normalize_obs else None

    #################################################
    # Helpers (given)
    #################################################

    def normalize_obs(self, obs, update=False):
        """Normalises observations with running statistics. Set update=True for training data only."""
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_rms is None:
            return obs
        if update:
            self.obs_rms.update(obs)
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10).astype(np.float32)

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def get_action(self, norm_obs, deterministic=False):
        """norm_obs: normalised observations (N, ob_dim). Returns numpy (actions, log_probs)."""
        action, log_prob = self.policy.get_action(self._t(norm_obs), deterministic)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    @torch.no_grad()
    def get_value(self, norm_obs):
        """Returns V(s) as numpy (N,), or zeros if there is no baseline."""
        if self.critic is None:
            return np.zeros(len(norm_obs), dtype=np.float32)
        return self.critic(self._t(norm_obs)).cpu().numpy()

    def set_learning_rate_fraction(self, frac):
        """Linearly anneals the learning rates (frac goes from 1 to 0 over training)."""
        for g in self.policy_optimizer.param_groups:
            g["lr"] = frac * self.alg.learning_rate
        if self.critic is not None:
            for g in self.critic.optimizer.param_groups:
                g["lr"] = frac * self.alg.critic_learning_rate

    #################################################
    # Advantage estimation
    #################################################

    def estimate_advantages(self, rewards, values, dones, last_value):
        """
        Returns (advantages, value_targets), each of shape (T, N).
          - Without a baseline the advantage is the discounted reward-to-go, and there are
            no value targets. Episodes still running at the end of the rollout are not
            bootstrapped, since there is no value function.
          - With a baseline, the returns are bootstrapped with V(s_T) and the advantage is
            R_t - V(s_t).
        """
        if not self.alg.use_baseline:
            returns = compute_discounted_returns(rewards, dones, np.zeros_like(last_value), self.alg.gamma)
            return returns, None
        returns = compute_discounted_returns(rewards, dones, last_value, self.alg.gamma)
        return compute_advantages(returns, values)

    #################################################
    # Training
    #################################################

    def update(self, batch):
        """
        batch: dict of flattened numpy arrays (batch = T * N):
            obs (batch, ob_dim) normalised, actions, log_probs, values, advantages, value_targets.
        """
        obs = self._t(batch["obs"])
        actions = self._t(batch["actions"])
        old_log_probs = self._t(batch["log_probs"])
        advantages = self._t(batch["advantages"])
        batch_size = obs.shape[0]

        if self.alg.rl_alg == "pg":
            # One gradient step on the whole batch: vanilla policy gradient.
            update_epochs, minibatch_size = 1, batch_size
        else:
            update_epochs = self.alg.update_epochs
            minibatch_size = max(batch_size // self.alg.num_minibatches, 1)

        logs = {"policy/loss": [], "policy/entropy": [], "policy/approx_kl": [], "policy/clipfrac": []}
        for epoch in range(update_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                mb_adv = advantages[idx]
                if self.alg.standardize_advantages and len(idx) > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_log_probs, entropy = self.policy.evaluate(obs[idx], actions[idx])

                # TODO: compute the policy loss.
                #   - rl_alg == "pg":  use pg_loss
                #   - rl_alg == "ppo": use ppo_clip_loss with self.alg.clip_coef
                #   Then subtract self.alg.ent_coef * entropy.mean() to get the total loss.
                #   Keep the probability ratio in a variable called `ratio`
                #   (for pg, use ratio = torch.exp(new_log_probs - old_log_probs[idx])).
                raise NotImplementedError

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.alg.max_grad_norm)
                self.policy_optimizer.step()

                with torch.no_grad():
                    log_ratio = torch.log(ratio)
                    # Low-variance KL estimator, see http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.alg.clip_coef).float().mean().item()
                logs["policy/loss"].append(policy_loss.item())
                logs["policy/entropy"].append(entropy.mean().item())
                logs["policy/approx_kl"].append(approx_kl)
                logs["policy/clipfrac"].append(clipfrac)

            if self.alg.target_kl is not None and approx_kl > self.alg.target_kl:
                break

        logs = {k: float(np.mean(v)) for k, v in logs.items()}

        if self.critic is not None:
            logs.update(self.critic.update(obs, self._t(batch["value_targets"]), self._t(batch["values"])))
        return logs

    def state_dict(self):
        state = {"policy": self.policy.state_dict()}
        if self.critic is not None:
            state["critic"] = self.critic.state_dict()
        if self.obs_rms is not None:
            state["obs_rms"] = {"mean": self.obs_rms.mean, "var": self.obs_rms.var, "count": self.obs_rms.count}
        return state
