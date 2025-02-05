import torch
import torch.nn as nn

class FeatureDiscrepancyBlock(nn.Module):
    """ Multi-Scale Feature Discrepancy Block (FDB) """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv_adjust = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, prior_features, current_features, prev_fdb=None):
        x = torch.cat([prior_features, current_features], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if prev_fdb is not None:
            x = x + self.conv_adjust(prev_fdb)  # Adjusting dimensions before adding
        return x