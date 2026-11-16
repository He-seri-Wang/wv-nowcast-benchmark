#!/usr/bin/env python3
"""Block-wise phase-correlation optical flow (numpy+FFT only, deterministic).

Motivation: cv2's optical-flow stack proved untrustworthy on the development
machine (Farneback silently returns all-zero flow on small-amplitude float32 and
garbage on real smooth WV imagery; DeepFlow gave 3.00px then 0.00px on the
identical known-shift input across processes). Phase correlation is the classic
feature tracker for geostationary imagery (cf. satellite AMV wind retrieval):
robust to global brightness change, works on smooth gradients.

API:
  estimate_flow(a0, a1, tile=64, stride=32, keep=0.3) -> (H, W, 2) float32
  warp(a1, flow, steps) -> advected field (forward, dst(x) = src(x - v))
"""
import numpy as np
import cv2  # only for remap/GaussianBlur — those primitives verified OK

def _fill(a, k=9):
    v = np.isfinite(a).astype(np.float32)
    m = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    num = cv2.blur(m, (k, k), borderType=cv2.BORDER_REPLICATE)
    den = cv2.blur(v, (k, k), borderType=cv2.BORDER_REPLICATE)
    f = num / np.maximum(den, 1e-6)
    f[den < 1e-6] = np.nanmedian(a)
    return f.astype(np.float32)

def _prep(a):
    f = _fill(a)
    med = np.nanmedian(a)
    f = np.where(f < med - 60, med, f)          # out-of-disk plateau -> median
    z = (f - f.mean()) / max(f.std(), 1e-6)
    return np.ascontiguousarray((z * 50).astype(np.float32))

def _tile_shift(t0, t1, max_shift=16):
    """Phase-correlation shift of one tile pair with subpixel peak fit.
    Returns (sx, sy, peak) or None."""
    h, w = t0.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    F0 = np.fft.rfft2(t0 * win)
    F1 = np.fft.rfft2(t1 * win)
    R = F0 * np.conj(F1)
    R /= np.maximum(np.abs(R), 1e-9)
    corr = np.fft.irfft2(R, s=t0.shape)
    # peak search within +-max_shift
    cy, cx = np.unravel_index(np.argmax(corr), corr.shape)
    peak = corr[cy, cx]
    def wrap(i, n): return i - n if i > n // 2 else i
    sy, sx = wrap(cy, h), wrap(cx, w)
    if abs(sx) > max_shift or abs(sy) > max_shift:
        return None
    # subpixel parabolic fit around the integer peak
    def para(c, im, ip):
        d = im - 2 * c + ip
        return 0.0 if abs(d) < 1e-9 else 0.5 * (im - ip) / d * -1
    dy = para(corr[cy, cx],
              corr[(cy - 1) % h, cx], corr[(cy + 1) % h, cx])
    dx = para(corr[cy, cx],
              corr[cy, (cx - 1) % w], corr[cy, (cx + 1) % w])
    # R = F0*conj(F1) peaks at -d (d = true feature displacement); return +d
    # so the field follows the standard convention (flow == feature velocity,
    # forward advection = warp with xs - v).
    return -(sx + dx), -(sy + dy), peak

def estimate_flow(a0, a1, tile=64, stride=32, keep=0.3):
    """Sparse per-tile phase-correlation vectors -> dense flow field.
    Tiles below the `keep` peak-quantile are dropped (unreliable)."""
    p0, p1 = _prep(a0), _prep(a1)
    H, W = p0.shape
    vecs = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            r = _tile_shift(p0[y:y+tile, x:x+tile], p1[y:y+tile, x:x+tile])
            if r is not None:
                vecs.append((x + tile/2, y + tile/2, r[0], r[1], r[2]))
    if not vecs:
        return np.zeros((H, W, 2), np.float32), 0
    peaks = np.array([v[4] for v in vecs])
    thr = np.quantile(peaks, keep)
    vecs = [v for v in vecs if v[4] >= thr]
    # scatter to dense grid + smooth (Gaussian over the vector field)
    fx = np.zeros((H, W), np.float32); fy = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for x, y, sx, sy, pk in vecs:
        fx[int(y), int(x)] = sx; fy[int(y), int(x)] = sy; wsum[int(y), int(x)] = 1
    # fill empty cells by nearest (indices from distance transform on zeros)
    if (wsum == 0).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(wsum == 0, return_distances=False,
                                     return_indices=True)
        fx, fy = fx[tuple(idx)], fy[tuple(idx)]
    s = 8
    fx = cv2.GaussianBlur(fx, (0, 0), s); fy = cv2.GaussianBlur(fy, (0, 0), s)
    return np.stack([fx, fy], -1).astype(np.float32), len(vecs)

def warp(a1, flow, steps):
    """Advect a1 forward `steps` x flow. NaNs propagate via remap."""
    h, w = a1.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    cur = a1.copy()
    for _ in range(steps):
        cur = cv2.remap(cur, xs - flow[..., 0], ys - flow[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cur
