import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.q_functions import DiscreteQFunction

# 1. Define your custom PyTorch Module
class MyCustomPyTorchModel(nn.Module):
    def __init__(self, observation_shape, feature_size, hdim):
        super().__init__()
        input_dim = observation_shape[0]
        self.feature_size = feature_size
        
        # Example: A simple 2-layer MLP with BatchNorm
        self.net = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, feature_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
    def get_feature_size(self):
        return self.feature_size

class MyCustomEncoderFactory(EncoderFactory):
    def __init__(self, feature_size, hdim):
        self.feature_size = feature_size
        self.hdim = hdim

    def create(self, observation_shape):
        # d3rlpy calls this method to get the network instance
        return MyCustomPyTorchModel(observation_shape, feature_size=self.feature_size, hdim=self.hdim)

    def get_params(self, deep=False):
        # Required for saving/loading the algorithm
        return {"feature_size": self.feature_size}
    
    def get_type(self):
        return "custom_encoder"
    

class MyCustomQFunction(DiscreteQFunction):
    """
    Custom Q-Function for Discrete Actions (DQN).
    It takes the features from the encoder and outputs Q-values.
    """
    def __init__(self, encoder, action_size, hdim):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        
        feature_size = encoder.get_feature_size()
        
        self.head = nn.Sequential(
            nn.Linear(feature_size, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, action_size)  # Output size must match action_size
        )

    def forward(self, x):
        h = self._encoder(x)
        return self.head(h)

    def compute_error(self, observations, actions, rewards, target_q, terminals, gamma=0.99, reduction="mean"):
        return super().compute_error(observations, actions, rewards, target_q, terminals, gamma, reduction)

    @property
    def encoder(self):
        return self._encoder

# 2. Define the Factory
class MyCustomQFunctionFactory(QFunctionFactory):
    def __init__(self, hdim):
        super().__init__()
        self.hdim = hdim

    # def create(self, encoder, action_size):
    #     return MyCustomQFunction(encoder, action_size, hdim=self.hdim)
    def create_discrete(self, encoder, hidden_size, action_size):
        return MyCustomQFunction(encoder, action_size, hdim=self.hdim)

    def get_params(self, deep=False):
        return {}

    def get_type(self):
        # Mandatory for d3rlpy v2+ serialization
        return "custom_q_func"