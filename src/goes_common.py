"""GOES utilities used by this benchmark: 2x2 spatial pooling and the ported
bidirectional selective state-space (BiSSM) temporal module.

Trimmed from the experiment-line commons to exactly what the WV scripts import
(geolocation helpers lived with other experiment lines and are not needed here).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pool2(a, k=2):
    """k x k mean pooling that ignores nothing — pair with valid-count logic
    at the call site (NaNs must be filled/masked first; see scripts/*decode*)."""
    h, w = a.shape
    h2, w2 = (h // k) * k, (w // k) * k
    return a[:h2, :w2].reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))


# ---------------- ported MS-BiTASP temporal selective SSM (lite) ----------------

class TemporalSelectiveStateSpace(nn.Module):
    """Bidirectional selective state-space temporal module (ported verbatim from
    the MS-BiTASP radar nowcasting codebase; autocast guard relaxed for device)."""

    def __init__(self, channels: int, state_dim: int = 4) -> None:
        super().__init__()
        self.channels, self.state_dim = channels, state_dim
        self.norm = nn.GroupNorm(8, channels)
        self.input_projection = nn.Conv2d(channels, channels * 4, 1)
        self.A_log = nn.Parameter(torch.zeros(channels, state_dim))
        self.C = nn.Parameter(torch.randn(channels, state_dim) * 0.02)
        self.D = nn.Parameter(torch.ones(channels))
        self.delta_bias = nn.Parameter(torch.full((channels,), -2.0))
        self.merge = nn.Sequential(nn.Conv3d(channels * 2, channels, 1), nn.GELU(), nn.Conv3d(channels, channels, 1))
        self.scale = nn.Parameter(torch.tensor(0.02))
        nn.init.zeros_(self.merge[-1].weight); nn.init.zeros_(self.merge[-1].bias)

    def _scan(self, u, delta, selection, reverse):
        batch, steps, channels, height, width = u.shape
        state = u.new_zeros(batch, channels, self.state_dim, height, width)
        transition = -torch.exp(self.A_log.float()).view(1, channels, self.state_dim, 1, 1)
        readout = self.C.float().view(1, channels, self.state_dim, 1, 1)
        skip = self.D.float().view(1, channels, 1, 1)
        outputs = {}
        order = range(steps - 1, -1, -1) if reverse else range(steps)
        for index in order:
            step = delta[:, index].unsqueeze(2)
            decay = torch.exp(step * transition)
            drive = step * selection[:, index].unsqueeze(2) * u[:, index].unsqueeze(2)
            state = decay * state + drive
            outputs[index] = (state * readout).sum(dim=2) + skip * u[:, index]
        return torch.stack([outputs[i] for i in range(steps)], dim=1)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm(x.float().flatten(0, 1))
            u, delta, selection, gate = self.input_projection(normalized).chunk(4, dim=1)
            u = F.silu(u).view(batch, steps, channels, height, width)
            delta = F.softplus(delta.view(batch, steps, channels, height, width) + self.delta_bias.view(1, 1, channels, 1, 1))
            selection = torch.sigmoid(selection).view(batch, steps, channels, height, width)
            gate = torch.sigmoid(gate).view(batch, steps, channels, height, width)
            fwd = self._scan(u, delta, selection, reverse=False)
            bwd = self._scan(u, delta, selection, reverse=True)
            merged = torch.cat((fwd, bwd), dim=2)
            merged = self.merge(merged.permute(0, 2, 1, 3, 4).contiguous())
            merged = merged.permute(0, 2, 1, 3, 4).contiguous() * gate
        return x + self.scale * merged
