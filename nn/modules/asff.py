import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.conv import Conv  # or use your own


class ASFF(nn.Module):
    def __init__(self, level=0, channels=None, out_channels=None):
        if channels is None:
            channels = [128, 256, 128]
        super().__init__()
        self.level = level
        self.channels = channels
        self.out_channels = channels[level] if out_channels is None else out_channels
        self.conv = Conv(sum(channels), self.out_channels, 3, 1)

    def forward(self, x):
        x0, x1, x2 = x
        if self.level == 0:
            r0 = x0
            r1 = F.interpolate(x1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
            r2 = F.interpolate(x2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        elif self.level == 1:
            r0 = F.interpolate(x0, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            r1 = x1
            r2 = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        else:
            r0 = F.interpolate(x0, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            r1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            r2 = x2

        fused = torch.cat([r0, r1, r2], dim=1)
        return self.conv(fused)
