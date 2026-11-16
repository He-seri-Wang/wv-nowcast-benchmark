#!/usr/bin/env python3
"""Fetch ABI-L1b-RadF C08 (6.2 um water vapor, 2 km, 10-min native) from the
noaa-goes16 public S3 bucket (no credentials), convert Rad->BT (inverse Planck
with the per-file coefficients), crop to the BOX4 4 km grid, delete raw files.

Days 155-175 (2024) — 21 days for the benchmark. ~35 MB/file download traffic,
~1 MB/frame kept.

Usage: python3 scripts/fetch_c08.py [--start 155 --end 175]
Requires the AWS CLI (`aws` on PATH); no account needed (--no-sign-request).
"""
import os, sys, glob, subprocess
import numpy as np, netCDF4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import C08_RAW, WV_DATA, BOX4, PAD
from src.goes_common import pool2

def decode(f):
    r0, r1, c0, c1 = BOX4
    R0, R1, C0, C1 = 2 * r0 - PAD, 2 * r1 + PAD, 2 * c0 - PAD, 2 * c1 + PAD
    nc = netCDF4.Dataset(f)
    rad = np.ma.filled(nc.variables["Rad"][R0:R1 + 1, C0:C1 + 1].astype(np.float64), np.nan)
    fk1 = float(nc.variables["planck_fk1"][:])
    fk2 = float(nc.variables["planck_fk2"][:])
    bc1 = float(nc.variables["planck_bc1"][:])
    bc2 = float(nc.variables["planck_bc2"][:])
    nc.close()
    # official ABI inverse Planck: BT = (fk2 / ln(fk1/Rad + 1) - bc1) / bc2
    bt = (fk2 / np.log(fk1 / np.maximum(rad, 1e-6) + 1.0) - bc1) / bc2
    bt = np.where(np.isfinite(rad) & (rad > 0) & (bt > 150) & (bt < 400), bt, np.nan)
    n = np.isfinite(bt).astype(np.float32)
    v = np.where(np.isfinite(bt), bt, 0.0)
    b4 = pool2(v) / np.maximum(pool2(n), 1e-6)
    b4 = np.where(pool2(n) < 0.5, np.nan, b4)
    return b4.astype(np.float32)

def slot_id(fname):
    return fname.split("_s")[-1][:13]             # YYYYDDDHHMMSS

def main():
    a = sys.argv
    start = int(a[a.index("--start") + 1]) if "--start" in a else 155
    end = int(a[a.index("--end") + 1]) if "--end" in a else 175
    os.makedirs(C08_RAW, exist_ok=True)
    for doy in range(start, end + 1):
        day_str = f"2024{doy:03d}"
        have = {f.split("_s")[-1][:13] for f in glob.glob(f"{WV_DATA}/*_s{day_str}*.npz")}
        for hh in range(24):
            # list the hour's keys once
            pfx = f"s3://noaa-goes16/ABI-L1b-RadF/2024/{doy:03d}/{hh:02d}/"
            try:
                keys = subprocess.run(["aws", "s3", "ls", pfx, "--no-sign-request"],
                                      capture_output=True, text=True, timeout=60).stdout
            except Exception as e:
                print(f"{day_str} h{hh:02d} LIST FAIL {e}", flush=True); continue
            files = [l.split()[-1] for l in keys.splitlines()
                     if "M6C08" in l and l.split()[-1].endswith(".nc")]
            for fn in files:
                sid = slot_id(fn)
                if sid[:7] != day_str: continue                    # start-of-scan day
                if sid in have: continue
                local = f"{C08_RAW}/{fn}"
                r = subprocess.run(["aws", "s3", "cp", pfx + fn, local,
                                    "--no-sign-request"], capture_output=True, timeout=300)
                if r.returncode != 0:
                    print(f"{fn} CP FAIL rc={r.returncode}: "
                          f"{r.stderr.decode()[:150]}", flush=True); continue
                try:
                    b4 = decode(local)
                    np.savez_compressed(f"{WV_DATA}/C08_{fn[:-3]}.npz", bt=b4)
                except Exception as e:
                    print(f"{fn} DECODE FAIL {repr(e)[:60]}", flush=True)
                os.remove(local)
        nday = len(glob.glob(f"{WV_DATA}/*_s{day_str}*.npz"))
        print(f"{day_str}: {nday} frames", flush=True)
    print("C08 FETCH DONE", flush=True)

if __name__ == "__main__":
    main()
