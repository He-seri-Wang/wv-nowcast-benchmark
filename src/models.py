"""Model zoo for the WV nowcast benchmark."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .goes_common import TemporalSelectiveStateSpace


class UNet(nn.Module):
    """Small 3-level encoder-decoder used for both rungs of the ladder.

    temporal=False (spatial rung): the 6 input frames are stacked channel-wise
    (B, T*C, H, W) and encoded in one shot.
    temporal=True (BiSSM rung): frames are embedded per-step, aggregated by the
    bidirectional selective SSM, and the last hidden state drives the decoder.
    """

    def __init__(self, cin, out_leads, ch=24, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.enc1 = nn.Conv2d(cin if not temporal else ch, ch, 3, padding=1)
        self.enc2 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(ch * 2, ch * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec = nn.Conv2d(ch + ch, ch, 3, padding=1)
        self.head = nn.Conv2d(ch, out_leads, 3, padding=1)
        self.ssm = TemporalSelectiveStateSpace(ch, 4) if temporal else None
        self.fuse = nn.Conv2d(cin, ch, 3, padding=1) if temporal else None

    def forward(self, x):                       # x (B, T, C, H, W)
        if self.temporal:
            B, T = x.shape[:2]
            z = self.fuse(x.flatten(0, 1)).view(B, T, -1, *x.shape[3:])
            z = self.ssm(z)[:, -1]
        else:
            z = F.gelu(self.enc1(x.flatten(1, 2)))
        e2 = F.gelu(self.enc2(z)); m = F.gelu(self.mid(e2))
        u = F.gelu(self.up(m)); d = F.gelu(self.dec(torch.cat([u, z], 1)))
        return self.head(d)                     # (B, out_leads, H, W) normalized
