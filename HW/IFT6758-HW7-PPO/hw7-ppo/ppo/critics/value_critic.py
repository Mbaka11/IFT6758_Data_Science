import numpy as np
import torch
from torch import nn

from ppo.policies.MLP_policy import build_network


def value_loss(new_values, value_targets, old_values=None, clip_coef=None):
    """
    Regression loss for the value function V_phi(s).

    Arguments (all torch tensors of shape (batch,)):
        new_values:    V_phi(s_t) under the current parameters phi.
        value_targets: the regression targets for V(s_t) (the returns computed by the agent).
        old_values:    V(s_t) predicted by the critic when the data was collected
                       (only used when clip_coef is not None).
        clip_coef:     if not None, use the clipped value loss from the PPO implementations:
                         V_clip = old_values + clip(new_values - old_values, -clip_coef, clip_coef)
                         loss   = 0.5 * mean( max( (new_values - targets)^2, (V_clip - targets)^2 ) )
    Returns:
        A scalar tensor. Without clipping this is 0.5 * mean squared error.
    """
    # TODO: implement the (optionally clipped) value-function regression loss.
    raise NotImplementedError


class ValueCritic(nn.Module):
    """
    A learned state-value function V_phi(s), used as the baseline for the policy gradient.

    The critic has its own network and its own optimizer, so it is trained separately
    from the policy.
    """

    def __init__(self, ob_shape, layer_sizes=(64, 64), activation="tanh", learning_rate=1e-3,
                 update_epochs=10, num_minibatches=32, max_grad_norm=0.5, clip_coef=None):
        super().__init__()
        self.net = build_network(ob_shape, 1, layer_sizes, activation, out_std=1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.max_grad_norm = max_grad_norm
        self.clip_coef = clip_coef

    def forward(self, obs):
        """Returns V(obs) with shape (batch,)."""
        return self.net(obs).squeeze(-1)

    def update(self, obs, value_targets, old_values):
        """
        Fit V_phi(s) to the value targets by minimising value_loss with minibatch SGD.

        Arguments (torch tensors):
            obs:           (batch, ob_dim) normalised observations.
            value_targets: (batch,) regression targets.
            old_values:    (batch,) critic predictions at collection time.
        Returns:
            dict of logging statistics.
        """
        batch_size = obs.shape[0]
        minibatch_size = max(batch_size // self.num_minibatches, 1)
        losses = []
        for _ in range(self.update_epochs):
            perm = torch.randperm(batch_size, device=obs.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                # TODO: one gradient step on the value function for this minibatch.
                #   1. predict V(s) for obs[idx]
                #   2. compute the loss with value_loss(...) (pass self.clip_coef)
                #   3. zero the gradients, backpropagate, clip the gradient norm to
                #      self.max_grad_norm (nn.utils.clip_grad_norm_) and take an optimizer step
                #   4. append loss.item() to losses
                raise NotImplementedError
        return {"critic/value_loss": float(np.mean(losses))}
