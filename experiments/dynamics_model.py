import torch
import torch.nn as nn
import numpy as np


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DynamicsModel(nn.Module):
    def __init__(self, observation_space_shape, num_actions):
        super(DynamicsModel, self).__init__()
        h, w, c = observation_space_shape
        num_actions  = num_actions[0]
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        
        self.feature_size = 64 * 4 * 4
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_actions)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + num_actions, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.feature_size)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        return self.inverse_net(torch.cat((state_ft, next_state_ft), 1)), self.forward_net(
            torch.cat((state_ft, action), 1)), next_state_ft
    
    
m = DynamicsModel((16, 16, 27), (334,))