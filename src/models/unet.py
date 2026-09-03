"""A compact time-conditioned UNet for predicting diffusion noise.

Original implementation (not adapted from a published codebase): sinusoidal
timestep embedding -> per-resolution residual blocks with the embedding
injected additively -> strided-conv down/up-sampling with skip connections ->
self-attention at the lowest resolution, where global context matters most
for reconstructing structured industrial parts.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding of integer timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_time_embedding(timesteps, self.dim))


class ResidualBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, with the time embedding added in between,
    plus a residual (1x1-projected if the channel count changes)."""

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(time_embed)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Single-head spatial self-attention, flattening H*W into the sequence dim."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, c, h * w) for t in (q, k, v))

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    """Predicts the noise added to an image at diffusion timestep t.

    channel_mults controls depth: e.g. (1, 2, 4) with base_channels=64 gives
    3 resolutions (64, 128, 256 channels), halving spatial size at each step.
    Self-attention is applied only at the lowest (bottleneck) resolution to
    keep memory/compute reasonable on a single consumer GPU.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        time_dim = time_embed_dim * 4
        self.time_embedding = TimeEmbedding(time_embed_dim)

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        for level, out_ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch, time_dim))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channels) - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResidualBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention2d(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, out_ch in list(enumerate(channels))[::-1]:
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch + skip_channels.pop(), out_ch, time_dim))
                ch = out_ch
            if level != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)
        self.num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embedding(timesteps)

        h = self.stem(x)
        skips = [h]
        block_idx = 0
        for level in range(len(self.downsamples)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, time_embed)
                block_idx += 1
                skips.append(h)
            h = self.downsamples[level](h)
            if not isinstance(self.downsamples[level], nn.Identity):
                skips.append(h)

        h = self.mid_block1(h, time_embed)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_embed)

        block_idx = 0
        for level in range(len(self.upsamples)):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.up_blocks[block_idx](h, time_embed)
                block_idx += 1
            h = self.upsamples[level](h)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
