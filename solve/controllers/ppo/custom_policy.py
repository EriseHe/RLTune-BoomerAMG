"""Custom PPO controller policy architecture."""

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class IdentityExtractor(BaseFeaturesExtractor):
    """Pass observations through unchanged."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self._features_dim = observation_space.shape[0]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.norm(self.act(self.fc(x)))


class CustomMLPExtractor(nn.Module):
    def __init__(self, input_dim: int, pi_dim: int, vf_dim: int, n_blocks: int):
        super().__init__()
        pi_blocks = [ResidualBlock(pi_dim) for _ in range(n_blocks)]
        vf_blocks = [ResidualBlock(vf_dim) for _ in range(n_blocks)]

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, pi_dim),
            nn.Tanh(),
            *pi_blocks,
        )
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, vf_dim),
            nn.Tanh(),
            *vf_blocks,
        )

        self.latent_dim_pi = pi_dim
        self.latent_dim_vf = vf_dim

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        pi_dim: int = 256,
        vf_dim: int = 128,
        n_blocks: int = 2,
        **kwargs
    ):
        self.pi_dim = pi_dim
        self.vf_dim = vf_dim
        self.n_blocks = n_blocks
        kwargs["features_extractor_class"] = IdentityExtractor
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMLPExtractor(
            input_dim=self.features_dim,
            pi_dim=self.pi_dim,
            vf_dim=self.vf_dim,
            n_blocks=self.n_blocks,
        )
