#!/usr/bin/env python3
"""Generate the repo's summary figures from results JSONs + a saved checkpoint.

figs/per_lead_rmse.png       v1 (10-60 min) and v2 (60-180 min) basins, all tiers
figs/v2_advantage_snapshot.png
                             test window where the v2 U-Net beats persistence by
                             the largest margin: input / truth / persistence / model
                             at +60, +120, +180 min

Usage: python scripts/make_figs.py [--gpu 1]
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.constants import WV_DATA, CKPT_DIR, RESULTS, CROP, TRAIN, VAL, TEST, bump
from src.models import UNet

FIGS = os.path.join(os.path.dirname(HERE), "figs")
os.makedirs(FIGS, exist_ok=True)

# ---------------- fig 1: per-lead curves ----------------
def fig_curves():
    v1 = json.load(open(f"{RESULTS}/pilot_wv.json"))["ladder"]
    v2 = json.load(open(f"{RESULTS}/pilot_wv2.json"))["ladder"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"persistence": ("k", "o", "persistence"),
              "pcflow": ("tab:blue", "s", "phase-correlation optical flow"),
              "unet_spatial": ("tab:red", "^", "U-Net (spatial)"),
              "unet_bissm": ("tab:orange", "v", "U-Net + BiSSM")}
    for ax, lad, lead_min, title in (
            (axes[0], v1, 10, "v1: 10-60 min basin"),
            (axes[1], v2, 20, "v2: 60-180 min basin")):
        for t in lad:
            c, m, lab = styles[t["model"]]
            ks = sorted(int(k) for k in t["per_lead_K"])
            ax.plot([k * lead_min for k in ks], [t["per_lead_K"][str(k)] for k in ks],
                    color=c, marker=m, ms=4, label=lab)
        ax.set_xlabel("lead time (min)")
        ax.set_ylabel("masked RMSE (K)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    a = next(t for t in v2 if t["model"] == "unet_spatial")["agg60_180_K"]
    axes[1].set_title(f"v2: 60-180 min basin\n"
                      f"U-Net vs persistence: +{a['gain']:.3f} K = {a['gain']/a['se']:.1f} se"
                      f" {'(PASS)' if a['pass_2se'] else '(fail)'}")
    fig.suptitle("GOES water-vapor channel nowcasting, test split", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/per_lead_rmse.png", dpi=150, bbox_inches="tight")
    print("-> figs/per_lead_rmse.png")

# ---------------- fig 2: v2 advantage snapshot ----------------
T_IN = 6
LEADS = [6, 8, 10, 12, 14, 16, 18]                     # x10 min -> 60..180
SPAN = 24
CROP_H, CROP_W = CROP

def fig_snapshot(dev):
    files = sorted(glob.glob(f"{WV_DATA}/*.npz"), key=lambda f: f.split("_s")[-1][:13])
    slots = {}
    for f in files:
        sid = f.split("_s")[-1][:13]
        if "2024155" <= sid[:7] <= "2024175":
            slots[sid] = np.load(f)["bt"]
    times = sorted(slots)
    runs, cur = [], [times[0]]
    for t in times[1:]:
        if bump(t, -10) == cur[-1]: cur.append(t)
        else: runs.append(cur); cur = [t]
    runs.append(cur)
    runs = [r for r in runs if len(r) >= SPAN]

    W = []
    for r in runs:
        for i in range(len(r) - SPAN + 1):
            win = r[i:i + SPAN]
            day = win[T_IN - 1][:7]
            xin = np.stack([slots[t] for t in win[:T_IN]])[:, :CROP_H, :CROP_W]
            yout = np.stack([slots[t] for t in [win[T_IN - 1 + L] for L in LEADS]])[:, :CROP_H, :CROP_W]
            W.append((day, win[T_IN - 1], xin.astype(np.float32), yout.astype(np.float32)))
    tr = [w for w in W if TRAIN[0] <= w[0] <= TRAIN[1]]
    te = [w for w in W if TEST[0] <= w[0] <= TEST[1]]
    mu = float(np.nanmean(np.concatenate([w[3].ravel() for w in tr[:200]])))
    sd = float(np.nanstd(np.concatenate([w[3].ravel() for w in tr[:200]])))

    def prep_x(w):
        x = (w[2] - mu) / sd
        return np.stack([np.nan_to_num(x), np.isfinite(x).astype(np.float32)], 1)

    model = UNet(2 * T_IN, len(LEADS), temporal=False).to(dev)
    model.load_state_dict(torch.load(f"{CKPT_DIR}/wv2_unet_spatial.pt", map_location=dev))
    model.eval()

    # score every test window: persistence vs model, mean RMSE over leads (K)
    best = None
    with torch.no_grad():
        for k, (day, tc, xin, yout) in enumerate(te):
            m = np.isfinite(yout)
            last = xin[-1]
            rmse_p = float(np.mean([np.sqrt(np.nanmean((last - yout[L])**2)) for L in range(len(LEADS))]))
            x = torch.tensor(np.stack([prep_x((day, tc, xin, yout))])).to(dev)
            p = model(x)[0].cpu().numpy() * sd + mu
            rmse_m = float(np.mean([np.sqrt(np.nanmean((p[L] - yout[L])**2)) for L in range(len(LEADS))]))
            if m.sum() < 200 * len(LEADS): continue
            gain = rmse_p - rmse_m
            if best is None or gain > best[0]:
                best = (gain, rmse_p, rmse_m, k)
    gain, rp, rm, k = best
    day, tc, xin, yout = te[k]
    print(f"advantage window: day={day} t={tc} persistence={rp:.2f}K model={rm:.2f}K gain={gain:.2f}K")

    x = torch.tensor(np.stack([prep_x(te[k])])).to(dev)
    with torch.no_grad():
        p = model(x)[0].cpu().numpy() * sd + mu
    cols = [0, 3, 6]                                     # +60, +120, +180 min
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    vmin, vmax = np.nanpercentile(yout[[0, 3, 6]], [2, 98])
    axes[0, 0].imshow(xin[-1], cmap="turbo_r", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"input (last frame)\nday {day} {tc[7:]}UTC")
    for j, L in enumerate(cols):
        lead = LEADS[L] * 10
        axes[0, j + 1].imshow(yout[L], cmap="turbo_r", vmin=vmin, vmax=vmax)
        axes[0, j + 1].set_title(f"truth +{lead} min")
        axes[1, j + 1].imshow(xin[-1], cmap="turbo_r", vmin=vmin, vmax=vmax)
        axes[1, j + 1].set_title(f"persistence +{lead} min")
        axes[2, j + 1].imshow(p[L], cmap="turbo_r", vmin=vmin, vmax=vmax)
        axes[2, j + 1].set_title(f"U-Net +{lead} min")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.05, 0.5, "persistence\n(last frame repeated)", fontsize=12,
                    ha="left", va="center")
    axes[2, 0].axis("off")
    axes[2, 0].text(0.05, 0.6, f"persistence mean RMSE {rp:.2f} K\nU-Net mean RMSE {rm:.2f} K\ngain {gain:.2f} K",
                    fontsize=12, va="center")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{FIGS}/v2_advantage_snapshot.png", dpi=150, bbox_inches="tight")
    print("-> figs/v2_advantage_snapshot.png")

if __name__ == "__main__":
    A = argparse.ArgumentParser()
    A.add_argument("--gpu", type=int, default=1)
    a = A.parse_args()
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    fig_curves()
    fig_snapshot(dev)
