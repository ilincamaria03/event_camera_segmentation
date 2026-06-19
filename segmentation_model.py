from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_c,
                in_c,
                3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=in_c,
                bias=False,
            ),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            norm(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.main = nn.Sequential(
            ConvBlock(in_c, out_c),
            ConvBlock(out_c, out_c, stride=2),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=2, bias=False),
            norm(out_c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class DecodeBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_c, out_c),
            ConvBlock(out_c, out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ContextBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c = max(out_c // 4, 8)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            norm(mid_c),
            nn.ReLU(inplace=True),
        )
        self.b2 = ConvBlock(in_c, mid_c, dilation=2)
        self.b3 = ConvBlock(in_c, mid_c, dilation=4)
        self.b4 = ConvBlock(in_c, mid_c, dilation=8)
        self.project = nn.Sequential(
            nn.Conv2d(mid_c * 4, out_c, 1, bias=False),
            norm(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return self.project(torch.cat(parts, dim=1))


class SegceptionLite(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 32, aux: bool = True):
        super().__init__()
        self.use_aux = aux

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1, bias=False),
            norm(base),
            nn.ReLU(inplace=True),
            ConvBlock(base, base),
        )

        self.enc1 = DownBlock(base, base * 2)
        self.enc2 = DownBlock(base * 2, base * 4)
        self.enc3 = DownBlock(base * 4, base * 8)
        self.enc4 = DownBlock(base * 8, base * 16)

        self.aspp = ContextBlock(base * 16, base * 8)

        self.skip3 = nn.Conv2d(base * 8, base * 4, 1, bias=False)
        self.skip2 = nn.Conv2d(base * 4, base * 2, 1, bias=False)
        self.skip1 = nn.Conv2d(base * 2, base, 1, bias=False)

        self.dec3 = DecodeBlock(base * 8 + base * 4, base * 4)
        self.dec2 = DecodeBlock(base * 4 + base * 2, base * 2)
        self.dec1 = DecodeBlock(base * 2 + base, base)

        self.aux_head = nn.Conv2d(base * 2, num_classes, 1)
        self.head = nn.Conv2d(base, num_classes, 1)

    @staticmethod
    def up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        y = self.aspp(x4)

        y = self.up(y, x3)
        y = self.dec3(torch.cat([y, self.skip3(x3)], dim=1))

        y = self.up(y, x2)
        y = self.dec2(torch.cat([y, self.skip2(x2)], dim=1))
        aux_out = F.interpolate(self.aux_head(y), size=size, mode="bilinear", align_corners=False)

        y = self.up(y, x1)
        y = self.dec1(torch.cat([y, self.skip1(x1)], dim=1))

        out = self.head(y)
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)

        return {"out": out, "aux": aux_out}


def segmentation_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    loss_fn: nn.Module,
    aux_weight: float = 0.4,
) -> torch.Tensor:
    loss = loss_fn(outputs["out"], target)
    if "aux" in outputs and outputs["aux"] is not None:
        loss = loss + aux_weight * loss_fn(outputs["aux"], target)
    return loss


def make_class_weights(pixel_counts, eps: float = 1e-6) -> torch.Tensor:
    n = torch.as_tensor(pixel_counts, dtype=torch.float32)
    f = n / n.sum().clamp_min(eps)
    w = 1.0 / torch.log(1.02 + f.clamp_min(eps))
    w[n <= 0] = 0.0

    ok = w > 0
    if ok.any():
        w[ok] = w[ok] / w[ok].mean()
    return w

def norm(ch: int) -> nn.Module:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)
