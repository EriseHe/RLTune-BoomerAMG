# paper_policy.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class IdentityExtractor(BaseFeaturesExtractor):
    """No feature engineering: pass obs straight into MLP."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self._features_dim = observation_space.shape[0]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


class PaperMLPExtractor(nn.Module):
    """
    Actor: 64 -> Drop(0.5) -> 128 -> Drop(0.5) -> 256 -> 512 -> Drop(0.5)
    Critic: 32 -> 32
    """
    def __init__(self, input_dim: int, act_drop_p=0.5):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=act_drop_p),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=act_drop_p),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=act_drop_p),
        )

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.latent_dim_pi = 512
        self.latent_dim_vf = 32

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class PaperPPOPolicy(ActorCriticPolicy):
    """
    - Dropout in actor
    - Tanh action head (matches paper diagram)
    - Separate lrs for actor and critic (matches paper table)
    """
    def __init__(
        self,
        *args,
        actor_lr=1e-5,
        critic_lr=5e-2,   # paper says 0.05
        act_drop_p=0.5,
        **kwargs
    ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.act_drop_p = act_drop_p

        # Use identity extractor so "tuple of states" goes directly into MLP
        kwargs["features_extractor_class"] = IdentityExtractor
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PaperMLPExtractor(
            input_dim=self.features_dim,
            act_drop_p=self.act_drop_p,
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)

        # Replace action head to include Tanh (paper diagram)
        action_dim = self.action_space.shape[0]
        self.action_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim),
            nn.Tanh(),
        )

        # Rebuild optimizer with two param groups
        self.optimizer = th.optim.Adam(
            [
                {"params": list(self.features_extractor.parameters()) +
                           list(self.mlp_extractor.policy_net.parameters()) +
                           list(self.action_net.parameters()),
                 "lr": self.actor_lr},
                {"params": list(self.mlp_extractor.value_net.parameters()) +
                           list(self.value_net.parameters()),
                 "lr": self.critic_lr},
            ]
        )