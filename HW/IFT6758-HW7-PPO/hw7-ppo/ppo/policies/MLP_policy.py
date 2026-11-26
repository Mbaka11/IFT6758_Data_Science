import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialisation, as used in most PPO implementations."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(in_dim, out_dim, layer_sizes, activation="tanh", out_std=1.0):
    """Builds an MLP: in_dim -> layer_sizes -> out_dim (no output activation)."""
    act = _ACTIVATIONS[activation]
    layers, last = [], in_dim
    for size in layer_sizes:
        layers += [layer_init(nn.Linear(last, size)), act()]
        last = size
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    """
    Small CNN for MinAtar observations of shape (H, W, C) (channels last), as in
    Young & Tian (2019): one 3x3 convolution with 16 filters, then a flattened output.
    """

    def __init__(self, ob_shape, channels=16):
        super().__init__()
        h, w, c = ob_shape
        self.conv = nn.Sequential(layer_init(nn.Conv2d(c, channels, kernel_size=3, stride=1)), nn.ReLU())
        self.out_dim = channels * (h - 2) * (w - 2)

    def forward(self, obs):
        return self.conv(obs.permute(0, 3, 1, 2)).flatten(1)


def build_network(ob_shape, out_dim, layer_sizes, activation="tanh", out_std=1.0):
    """
    Builds the network used by the policy and the critic.
    - Vector observations (e.g. HalfCheetah): an MLP.
    - Image observations of shape (H, W, C) (e.g. MinAtar): a CNN followed by an MLP.
    """
    ob_shape = (ob_shape,) if isinstance(ob_shape, int) else tuple(ob_shape)
    if len(ob_shape) == 1:
        return build_mlp(ob_shape[0], out_dim, layer_sizes, activation, out_std)
    encoder = ImageEncoder(ob_shape)
    return nn.Sequential(encoder, build_mlp(encoder.out_dim, out_dim, layer_sizes, activation, out_std))


class MLPPolicy(nn.Module):
    """
    Stochastic policy pi(a|s).

    - Discrete action spaces: a Categorical distribution over the logits of the MLP.
    - Continuous action spaces: a diagonal Gaussian whose mean is the output of the MLP
      and whose log standard deviation is a state-independent learned parameter.

    This class is given to you. You do not need to modify it.
    """

    def __init__(self, ob_shape, ac_dim, discrete, layer_sizes=(64, 64), activation="tanh", init_log_std=0.0):
        super().__init__()
        self.discrete = discrete
        self.net = build_network(ob_shape, ac_dim, layer_sizes, activation, out_std=0.01)
        if not discrete:
            self.log_std = nn.Parameter(torch.ones(1, ac_dim) * init_log_std)

    def forward(self, obs):
        """Returns the action distribution pi(.|obs)."""
        out = self.net(obs)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, torch.exp(self.log_std.expand_as(out)))

    def log_prob(self, dist, actions):
        """log pi(a|s), summed over action dimensions for continuous actions. Shape: (batch,)"""
        if self.discrete:
            return dist.log_prob(actions.long())
        return dist.log_prob(actions).sum(-1)

    def entropy(self, dist):
        """Entropy of pi(.|s), summed over action dimensions. Shape: (batch,)"""
        if self.discrete:
            return dist.entropy()
        return dist.entropy().sum(-1)

    def get_action(self, obs, deterministic=False):
        """Samples an action and returns (action, log_prob)."""
        dist = self(obs)
        if deterministic:
            action = dist.probs.argmax(-1) if self.discrete else dist.mean
        else:
            action = dist.sample()
        return action, self.log_prob(dist, action)

    def evaluate(self, obs, actions):
        """Returns (log_prob, entropy) of the given actions under the current policy."""
        dist = self(obs)
        return self.log_prob(dist, actions), self.entropy(dist)
