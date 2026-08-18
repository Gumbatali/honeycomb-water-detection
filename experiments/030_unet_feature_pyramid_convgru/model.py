"""Frozen U-Net feature-pyramid fusion with a full-frame temporal ConvGRU."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FrozenUNetPyramid(nn.Module):
    """The encoder portion of the validated 7-channel U-Net checkpoint."""
    def __init__(self) -> None:
        super().__init__()
        self.e1, self.e2 = DoubleConv(7, 32), DoubleConv(32, 64)
        self.e3, self.e4 = DoubleConv(64, 128), DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        return e2, e3, e4

    def load_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        own.update({key: value for key, value in state_dict.items() if key in own})
        self.load_state_dict(own)
        for parameter in self.parameters(): parameter.requires_grad_(False)
        self.eval()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False), nn.GroupNorm(8, out_channels), nn.GELU(),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.GroupNorm(8, out_channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.main(x) + self.skip(x))


class GatedFusion(nn.Module):
    """Inject a static U-Net feature map only where the thermal feature requests it."""
    def __init__(self, channels: int, unet_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(unet_channels, channels, 1)
        self.gate = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, thermal: torch.Tensor, unet: torch.Tensor) -> torch.Tensor:
        unet = F.interpolate(self.project(unet), size=thermal.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat((thermal, unet), dim=1)))
        return thermal + gate * unet


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((x, state), 1)).chunk(2, 1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * state), 1)))
        return (1 - update) * state + update * candidate


class UNetFeatureConvGRU(nn.Module):
    """Full thermal video + frozen U-Net encoder features -> semantic water map."""
    def __init__(self, hidden: int = 128, dropout: float = 0.10, num_classes: int = 7) -> None:
        super().__init__()
        self.unet = FrozenUNetPyramid()
        self.s1 = nn.Sequential(ResidualBlock(1, 48, 2), ResidualBlock(48, 48))
        self.s2 = nn.Sequential(ResidualBlock(48, 96, 2), ResidualBlock(96, 96))
        self.s3 = nn.Sequential(ResidualBlock(96, hidden, 2), ResidualBlock(hidden, hidden))
        self.f1, self.f2, self.f3 = GatedFusion(48, 64), GatedFusion(96, 128), GatedFusion(hidden, 256)
        self.temporal, self.dropout = ConvGRUCell(hidden), nn.Dropout2d(dropout)
        self.decode2 = ResidualBlock(hidden + 96, 96)
        self.decode1 = ResidualBlock(96 + 48, 64)
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, thermal: torch.Tensor, unet_input: torch.Tensor) -> torch.Tensor:
        """thermal B,T,1,240,320; unet_input B,7,480,640 after documented preprocessing."""
        batch, steps, _, height, width = thermal.shape
        with torch.no_grad(): u1, u2, u3 = self.unet(unet_input)
        x = thermal.reshape(batch * steps, 1, height, width)
        s1 = self.s1(x); s1 = self.f1(s1, u1.repeat_interleave(steps, 0))
        s2 = self.s2(s1); s2 = self.f2(s2, u2.repeat_interleave(steps, 0))
        s3 = self.s3(s2); s3 = self.f3(s3, u3.repeat_interleave(steps, 0))
        s1 = s1.reshape(batch, steps, 48, *s1.shape[-2:]); s2 = s2.reshape(batch, steps, 96, *s2.shape[-2:]); s3 = s3.reshape(batch, steps, -1, *s3.shape[-2:])
        state = torch.zeros_like(s3[:, 0])
        for time in range(steps): state = self.temporal(s3[:, time], state)
        d2 = F.interpolate(self.dropout(state), scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.decode2(torch.cat((d2, s2.mean(1)), 1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.decode1(torch.cat((d1, s1.mean(1)), 1))
        return self.out(F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False))

    def train(self, mode: bool = True) -> "UNetFeatureConvGRU":
        """Keep frozen BatchNorm statistics fixed even while the outer model trains."""
        super().train(mode)
        self.unet.eval()
        return self
