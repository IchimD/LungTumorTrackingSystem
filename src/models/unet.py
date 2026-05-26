import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.inc   = ConvBlock(in_channels, base_channels)
        self.down1 = Down(base_channels,     base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=0.1)
        self.down4 = Down(base_channels * 8, base_channels * 8, dropout=0.3)  # bottleneck

        self.up1 = Up(base_channels * 16, base_channels * 4, bilinear=bilinear, dropout=0.3)
        self.up2 = Up(base_channels * 8,  base_channels * 2, bilinear=bilinear, dropout=0.1)
        self.up3 = Up(base_channels * 4,  base_channels,     bilinear=bilinear)
        self.up4 = Up(base_channels * 2,  base_channels,     bilinear=bilinear)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
