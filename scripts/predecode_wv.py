#!/usr/bin/env python3
"""Predecode ABI-L2-MCMIPF C08 (6.2 um upper-level water vapor BT, 2 km) into
4 km BOX4 npz frames. Use this when you already have MCMIPF .nc files locally;
use fetch_c08.py to download + decode the L1b 10-min archive from scratch.

Usage: python3 scripts/predecode_wv.py
"""
import os, sys, glob as _glob
import numpy as np, netCDF4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import GOES_RAW, WV_DATA, BOX4, PAD
from src.goes_common import pool2

def decode(f):
    r0, r1, c0, c1 = BOX4
    R0, R1, C0, C1 = 2 * r0 - PAD, 2 * r1 + PAD, 2 * c0 - PAD, 2 * c1 + PAD
    nc = netCDF4.Dataset(f)
    bt = np.ma.filled(nc.variables["CMI_C08"][R0:R1 + 1, C0:C1 + 1].astype(np.float32), np.nan)
    nc.close()
    # physical guard: clear-sky BT 180-340 K, cloudy tops can reach ~170 K
    bt = np.where(np.isfinite(bt) & (bt > 150) & (bt < 400), bt, np.nan)
    n = np.isfinite(bt).astype(np.float32)
    v = np.where(np.isfinite(bt), bt, 0.0)
    b4 = pool2(v) / np.maximum(pool2(n), 1e-6)
    b4 = np.where(pool2(n) < 0.5, np.nan, b4)          # need >=2 of 4 subpixels
    return b4.astype(np.float32)

def main():
    os.makedirs(WV_DATA, exist_ok=True)
    files = sorted(_glob.glob(f"{GOES_RAW}/*MCMIPF*.nc"),
                   key=lambda f: f.split("_s")[-1][:13])
    todo = [f for f in files if not os.path.exists(
        f"{WV_DATA}/{os.path.basename(f).replace('.nc', '.npz')}")]
    print(f"{len(files)} raw, {len(todo)} to decode", flush=True)
    for i, f in enumerate(todo):
        out = f"{WV_DATA}/{os.path.basename(f).replace('.nc', '.npz')}"
        try:
            b4 = decode(f)
            np.savez_compressed(out, bt=b4)
            if (i + 1) % 40 == 0:
                print(f"{i+1}/{len(todo)} finite={np.isfinite(b4).mean():.3f} "
                      f"BT mean={np.nanmean(b4):.1f}", flush=True)
        except Exception as e:
            print("FAIL", os.path.basename(f), repr(e)[:80], flush=True)
    print("WV PREDECODE DONE", flush=True)

if __name__ == "__main__":
    main()
