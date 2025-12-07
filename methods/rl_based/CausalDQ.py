# https://arxiv.org/abs/2507.09742

import dataclasses
import d3rlpy
import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
from d3rlpy.algos import DQN
from d3rlpy.algos.qlearning.torch.dqn_impl import DQNImpl
from d3rlpy.base import LearnableConfig, DeviceArg
from d3rlpy.optimizers.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.models.q_functions import QFunctionFactory, make_q_func_field


class CausalDQNImpl(DQNImpl):
    """
    Implementation logic for Causal DQ.
    Overrides the loss computation to add Causal Entropy Regularization.
    """
    def __init__(self, alpha: float, mask_size: int, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._mask_size = mask_size

    def compute_loss(self, batch, q_tpn, targ_q_tpn, gamma):
        # 1. Compute the Standard Bellman Error (TD Loss)
        # We let the parent class handle the Double DQN logic
        td_loss = super().compute_loss(batch, q_tpn, targ_q_tpn, gamma)

        # 2. Compute Causal Entropy Regularization
        # Assumption: The 'Causal Mask' is appended to the end of the observation
        obs_t = batch.observations
        
        # Extract the mask (Last 'mask_size' columns)
        causal_mask = obs_t[:, -self._mask_size:]
        
        # Get Q-values from the online network
        q_values = self._q_func(obs_t)
        
        # Calculate policy distribution (Softmax)
        probs = F.softmax(q_values, dim=1)
        log_probs = F.log_softmax(q_values, dim=1)
        
        # Calculate Entropy specifically on Causal Paths
        # H_causal = - sum( Mask * p * log(p) )
        masked_entropy = -torch.sum(causal_mask * probs * log_probs, dim=1).mean()

        # 3. Final Loss = TD Loss - (Alpha * Causal Entropy)
        # We subtract because we want to MAXIMIZE entropy (minimize negative entropy)
        total_loss = td_loss - (self._alpha * masked_entropy)

        return total_loss


class CausalDQN(DQN):
    """
    The Causal DQ Algorithm.
    Extends standard DQN to allow for causal entropy regularization.
    """
    def __init__(self, alpha=0.1, mask_size=10, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._mask_size = mask_size

    def _create_impl(self, dataset, observation_shape, action_size):
        # This connects our custom math (Impl) with this configuration class
        return CausalDQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            gamma=self._gamma,
            use_gpu=self._use_gpu,
            encoder_factory=self._encoder_factory, #override for custom state-to-action encoding
            q_func_factory=self._q_func_factory,
            optimizer_factory=self._optimizer_factory,
            target_update_interval=self._target_update_interval,
            # Custom params
            alpha=self._alpha,
            mask_size=self._mask_size,
        )
    

@dataclasses.dataclass()
class CausalDQNConfig(LearnableConfig):

    batch_size: int = 32
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000
    alpha:float = 0.1
    mask_size:int = 10

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "CausalDQN":
        return CausalDQN(config=self, 
                         device=device, 
                         enable_ddp=enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "causal_dqn"








