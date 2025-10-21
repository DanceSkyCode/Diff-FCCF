import torch
import math
import torch.nn.functional as F
import os
import cv2
import time
import torch.nn as nn
import numpy as np
from utils import tensor2img
import model.loss as loss
from torch_scatter import scatter_mean 
from torch.nn import LayerNorm
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn.functional as F
from CherbGraph import HyperEdgeGCN
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = HyperEdgeGCN()
        self.action_head = nn.Sequential(
            nn.Linear(256*256, 512),
            nn.ReLU(),
            nn.Linear(512, 256*256),
            nn.Tanh()
        )
    def forward(self, state):
        features = self.gcn(state)
        actions = self.action_head(features.flatten())
        return actions.view_as(state) * 0.1
class AdaptiveBoundary:
    def __init__(self, init_value=0.1):
        self.boundary = init_value
        self.history = []
    def update(self, current_distance):
        self.history.append(current_distance)
        if len(self.history) > 100:
            trend = np.polyfit(range(100), self.history[-100:], 1)[0]
            self.boundary *= (1 + 0.1*np.sign(trend))    
class RLTrainingWrapper:
    def __init__(self, device):
        self.policy_net = PolicyNetwork().to(device)
        self.optimizer_rl = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.boundary_controller = AdaptiveBoundary()

    def apply_rl_step(self, crcb, y_channel, original_crcb):
        state = crcb.detach().clone()
        original = original_crcb.detach().clone()
        action = self.policy_net(state)
        new_state = torch.clamp(state + action, 0, 1)
        with torch.no_grad():
            reward = self.calculate_reward(new_state, y_channel, original)
        advantage = reward - self.value_net(state)
        policy_loss = -(advantage * torch.log(action.probs)).mean()
        entropy_loss = -self.entropy_coef * action.entropy().mean()
        self.optimizer_rl.zero_grad()
        (policy_loss + entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer_rl.step()
        return new_state.detach()
