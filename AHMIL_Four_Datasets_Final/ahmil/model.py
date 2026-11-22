"""AHMIL model: MSIFE, FECA (FERA + PPEG), and Lorentz HDM."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: Optional[int] = None,
                 dropout: float = 0.25):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleInstanceEncoder(nn.Module):
    """Three parallel temporal branches with kernels 3, 11, and 25."""

    def __init__(self, input_channels: int, embed_dim: int = 64,
                 input_layout: str = "time_first", dropout: float = 0.5):
        super().__init__()
        if input_layout not in {"time_first", "channels_first"}:
            raise ValueError("input_layout must be 'time_first' or 'channels_first'.")
        self.input_channels = input_channels
        self.input_layout = input_layout
        self.feature_dim = embed_dim

        def branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                TemporalBlock(input_channels, 16, kernel_size),
                TemporalBlock(16, 32, 3, stride=2, padding=1),
                nn.MaxPool1d(2),
            )

        self.branch_3 = branch(3)
        self.branch_11 = branch(11)
        self.branch_25 = branch(25)
        self.fusion = nn.Sequential(
            nn.Conv1d(96, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D instance batch, got {tuple(x.shape)}")

        if self.input_layout == "time_first":
            if x.shape[-1] != self.input_channels:
                raise ValueError(
                    f"Expected {self.input_channels} EEG channels in the last axis, "
                    f"got {tuple(x.shape)}"
                )
            x = x.transpose(1, 2)
        elif x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels in axis 1, got {tuple(x.shape)}"
            )

        x = torch.cat(
            (self.branch_3(x), self.branch_11(x), self.branch_25(x)), dim=1
        )
        x = self.fusion(x)
        return self.pool(x).squeeze(-1)


class GaussianFuzzyLayer(nn.Module):
    """Channel-wise Gaussian membership, rule recombination, and residual enhancement."""

    def __init__(self, channels: int, num_rules: int = 3):
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be positive.")
        self.channels = channels
        self.num_rules = num_rules
        self.centers = nn.Parameter(torch.randn(channels, num_rules) * 0.1)
        self.scales_raw = nn.Parameter(torch.ones(channels, num_rules))
        self.rule_weights = nn.Parameter(torch.randn(channels, num_rules) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"GaussianFuzzyLayer expects [B, L, {self.channels}], got {tuple(x.shape)}"
            )
        scales = F.softplus(self.scales_raw).clamp_min(1e-6)
        membership = torch.exp(
            -((x.unsqueeze(-1) - self.centers[None, None, :, :]) /
              scales[None, None, :, :]).square()
        )
        fuzzy_component = (
            membership * self.rule_weights[None, None, :, :]
        ).sum(dim=-1)
        return x + self.alpha * fuzzy_component


class ParameterMatchedMLP(nn.Module):
    """Residual MLP control with the same parameter count as the fuzzy layer."""

    def __init__(self, channels: int, num_rules: int = 3):
        super().__init__()
        if num_rules < 1 or (3 * num_rules - 1) % 2:
            raise ValueError("Parameter matching requires a positive odd rule count.")
        hidden = (3 * num_rules - 1) // 2
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.gain = nn.Parameter(torch.full((channels,), 0.1))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(F.gelu(self.fc1(x)))
        return x + self.gain * residual + self.bias


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size: int, head_size: int, gamma: float):
        super().__init__()
        self.gamma = float(gamma)
        self.query = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.key = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.value = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        batch, length, _ = x.shape
        q = x @ self.query
        k = x @ self.key
        v = x @ self.value
        scores = q @ k.transpose(1, 2)

        positions = torch.arange(length, device=x.device)
        decay = self.gamma ** (positions[:, None] - positions[None, :]).abs()
        scores = scores.masked_fill(
            ~valid_mask[:, None, :], torch.finfo(scores.dtype).min
        ) if valid_mask is not None else scores
        weights = torch.softmax(scores, dim=-1) * decay.unsqueeze(0)
        return weights @ v, weights


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size: int, heads: int = 8,
                 enhancement: str = "fuzzy", num_rules: int = 3):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by the number of heads.")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_size = hidden_size // heads
        self.enhancement = enhancement
        gamma_values = (
            1.0 - torch.exp(
                torch.linspace(math.log(1.0 / 32.0), math.log(1.0 / 512.0), heads)
            )
        ).tolist()
        self.retention_heads = nn.ModuleList([
            SimpleRetention(hidden_size, self.head_size, gamma)
            for gamma in gamma_values
        ])
        self.gate = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.output = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)

        if enhancement == "fuzzy":
            self.enhancer = GaussianFuzzyLayer(hidden_size, num_rules)
        elif enhancement == "mlp":
            self.enhancer = ParameterMatchedMLP(hidden_size, num_rules)
        elif enhancement == "none":
            self.enhancer = nn.Identity()
        else:
            raise ValueError(f"Unknown enhancement mode: {enhancement}")

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        enhanced = self.enhancer(x)
        head_outputs = []
        head_weights = []
        for head in self.retention_heads:
            output, weights = head(enhanced, valid_mask)
            head_outputs.append(output)
            head_weights.append(weights.unsqueeze(1))
        context = torch.cat(head_outputs, dim=-1)
        gate = F.silu(x @ self.gate)
        output = (gate * context) @ self.output
        return output, torch.cat(head_weights, dim=1)


class FERA(nn.Module):
    def __init__(self, dim: int, heads: int = 8,
                 enhancement: str = "fuzzy", num_rules: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.retention = MultiScaleRetention(dim, heads, enhancement, num_rules)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor):
        context, weights = self.retention(self.norm(x), valid_mask)
        return x + context, weights


class PPEG(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv7 = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.conv5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, :1]
        instances = x[:, 1:].transpose(1, 2)
        enhanced = instances + self.conv7(instances) + self.conv5(instances) + self.conv3(instances)
        return torch.cat((cls_token, enhanced.transpose(1, 2)), dim=1)


class LorentzPrototypeClassifier(nn.Module):
    """Lorentz hyperbolic classifier with one learnable prototype per class."""

    def __init__(self, in_features: int, num_classes: int,
                 curvature: float = 1.0, scale_init: float = 2.5):
        super().__init__()
        if curvature <= 0:
            raise ValueError("curvature must be positive.")
        self.curvature = float(curvature)
        self.tangent_projection = nn.Linear(in_features, in_features)
        self.prototype_spatial = nn.Parameter(torch.randn(num_classes, in_features) * 0.01)
        self.scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(scale_init))))
        nn.init.eye_(self.tangent_projection.weight)
        nn.init.zeros_(self.tangent_projection.bias)

    def exp_map_zero(self, tangent: torch.Tensor) -> torch.Tensor:
        sqrt_c = math.sqrt(self.curvature)
        norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-8)
        scaled_norm = sqrt_c * norm
        time = torch.cosh(scaled_norm) / sqrt_c
        spatial = torch.sinh(scaled_norm) / (sqrt_c * norm) * tangent
        return torch.cat((time, spatial), dim=-1)

    def prototypes_on_manifold(self) -> torch.Tensor:
        sqrt_c = math.sqrt(self.curvature)
        time = torch.sqrt(
            1.0 / self.curvature + self.prototype_spatial.square().sum(dim=-1, keepdim=True)
        )
        return torch.cat((time, self.prototype_spatial), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.exp_map_zero(self.tangent_projection(x))
        prototypes = self.prototypes_on_manifold()
        inner = (
            -embedding[:, :1] @ prototypes[:, :1].transpose(0, 1)
            + embedding[:, 1:] @ prototypes[:, 1:].transpose(0, 1)
        )
        argument = (-self.curvature * inner).clamp_min(1.0 + 1e-7)
        distance = torch.acosh(argument) / math.sqrt(self.curvature)
        scale = F.softplus(self.scale_raw)
        return -scale * distance


class EuclideanPrototypeClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale_init: float = 2.5):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features) * 0.01)
        self.scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(scale_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(x, self.prototypes, p=2).square()
        return -F.softplus(self.scale_raw) * distance


class AHMIL(nn.Module):
    """Full model and controlled ablations defined in the manuscript."""

    VARIANTS = {"baseline", "mlp", "fuzzy", "hyper", "fuzzy_epc", "full"}

    def __init__(self, input_channels: int, num_classes: int,
                 input_layout: str = "time_first", embed_dim: int = 64,
                 num_rules: int = 3, curvature: float = 1.0,
                 variant: str = "full", num_heads: int = 8):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        self.variant = variant
        self.encoder = MultiScaleInstanceEncoder(
            input_channels=input_channels,
            embed_dim=embed_dim,
            input_layout=input_layout,
        )
        self.instance_projection = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, embed_dim),
            nn.ReLU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if variant in {"fuzzy", "fuzzy_epc", "full"}:
            enhancement = "fuzzy"
        elif variant == "mlp":
            enhancement = "mlp"
        else:
            enhancement = "none"

        self.fera1 = FERA(embed_dim, num_heads, enhancement, num_rules)
        self.ppeg = PPEG(embed_dim)
        self.fera2 = FERA(embed_dim, num_heads, enhancement, num_rules)
        self.final_norm = nn.LayerNorm(embed_dim)

        if variant in {"hyper", "full"}:
            self.classifier = LorentzPrototypeClassifier(
                embed_dim, num_classes, curvature=curvature
            )
        elif variant == "fuzzy_epc":
            self.classifier = EuclideanPrototypeClassifier(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, bags: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        if bags.ndim not in {4, 5}:
            raise ValueError(
                "AHMIL expects [batch, instances, time, channels] or "
                "[batch, instances, bands, electrodes]."
            )
        batch_size, num_instances = bags.shape[:2]
        instances = bags.reshape(-1, *bags.shape[2:])
        encoded = self.encoder(instances)
        encoded = self.instance_projection(encoded)
        encoded = encoded.reshape(batch_size, num_instances, -1)

        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls, encoded), dim=1)
        if mask is None:
            valid_mask = torch.ones(
                batch_size, num_instances + 1, dtype=torch.bool, device=bags.device
            )
        else:
            valid_mask = torch.cat(
                (torch.ones(batch_size, 1, dtype=torch.bool, device=bags.device), mask.bool()),
                dim=1,
            )

        sequence, _ = self.fera1(sequence, valid_mask)
        sequence = self.ppeg(sequence)
        sequence, attention = self.fera2(sequence, valid_mask)
        bag_embedding = self.final_norm(sequence)[:, 0]
        return self.classifier(bag_embedding), attention
