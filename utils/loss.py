import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, lamb=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lamb = lamb
        
    def forward(self, rgb, rgb_gt, noise_map, noise_map_gt):
        l_rgb = self.mse(rgb, rgb_gt)
        l_nm = self.mse(noise_map, noise_map_gt)
        return l_rgb + self.lamb * l_nm
