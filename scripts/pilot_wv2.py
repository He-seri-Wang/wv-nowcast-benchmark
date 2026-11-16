#!/usr/bin/env python3
"""WV nowcasting pilot v2 — re-aimed at the 60-180 min horizon where the probe
showed zero-training methods have no skill (pcflow crosses below persistence
at ~120 min).

Same data/split/budget as v1; targets = leads {6,8,10,12,14,16,18} x 10 min.
Kill bar (pre-declared): U-Net+BiSSM must beat persistence by >= 2 x day-block
bootstrap se on the full 60-180 min aggregate. pcflow reported alongside.

Usage: python3 scripts/pilot_wv2.py [--steps 3000] [--gpu 0] [--eval-only]
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import WV_DATA, CKPT_DIR, RESULTS, TRAIN, VAL, TEST, CROP, bump
from src.eval import agg, day_boot
from src.models import UNet
from src.pcflow import estimate_flow, warp as pc_warp

T_IN = 6
LEADS = [6, 8, 10, 12, 14, 16, 18]                     # x10 min -> 60..180
SPAN = 24                                              # frames per window (0..23)
CROP_H, CROP_W = CROP

def load_runs():
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
    return slots, [r for r in runs if len(r) >= SPAN]

def main():
    A = argparse.ArgumentParser()
    A.add_argument("--steps", type=int, default=3000)
    A.add_argument("--batch", type=int, default=8)
    A.add_argument("--gpu", type=int, default=0)
    A.add_argument("--out", default=os.path.join(RESULTS, "pilot_wv2.json"))
    A.add_argument("--eval-only", action="store_true")
    a = A.parse_args()
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"

    slots, runs = load_runs()
    W = []
    for r in runs:
        for i in range(len(r) - SPAN + 1):
            win = r[i:i + SPAN]
            day = win[T_IN - 1][:7]
            xin = np.stack([slots[t] for t in win[:T_IN]])[:, :CROP_H, :CROP_W]
            yout = np.stack([slots[t] for t in [win[T_IN - 1 + L] for L in LEADS]])[:, :CROP_H, :CROP_W]
            W.append((day, win[T_IN - 1], xin.astype(np.float32), yout.astype(np.float32)))
    tr = [w for w in W if TRAIN[0] <= w[0] <= TRAIN[1]]
    va = [w for w in W if VAL[0] <= w[0] <= VAL[1]]
    te = [w for w in W if TEST[0] <= w[0] <= TEST[1]]
    print(f"windows: train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)
    mu = float(np.nanmean(np.concatenate([w[3].ravel() for w in tr[:200]])))
    sd = float(np.nanstd(np.concatenate([w[3].ravel() for w in tr[:200]])))
    print(f"train stats: mu={mu:.2f}K sd={sd:.2f}K", flush=True)

    def prep_x(w):
        x = (w[2] - mu) / sd
        return np.stack([np.nan_to_num(x), np.isfinite(x).astype(np.float32)], 1)
    def prep_y(w):
        y = (w[3] - mu) / sd
        return np.nan_to_num(y, nan=0.0), np.isfinite(y)

    Xtr = torch.tensor(np.stack([prep_x(w) for w in tr]))
    Ytr = torch.tensor(np.stack([prep_y(w)[0] for w in tr]))
    Mtr = torch.tensor(np.stack([prep_y(w)[1] for w in tr]))
    Xva = torch.tensor(np.stack([prep_x(w) for w in va]))
    Yva = torch.tensor(np.stack([prep_y(w)[0] for w in va]))
    Mva = torch.tensor(np.stack([prep_y(w)[1] for w in va]))
    K = len(LEADS)

    def masked_rmse(p, y, m):
        d = torch.where(m, (p - y) ** 2, torch.zeros_like(p))
        return float((d.sum() / m.sum()).sqrt() * sd)

    def run_model(name, temporal, cin, batch, accum):
        torch.manual_seed(0)                                # seeded for reproducibility
        model = UNet(cin, K, temporal=temporal).to(dev)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        n = len(Xtr); best = (1e9, None)
        for st in range(a.steps):
            opt.zero_grad(); tot = 0.0
            for _ in range(accum):
                i = torch.randint(0, n, (batch,))
                xb, yb, mb = Xtr[i].to(dev), Ytr[i].to(dev), Mtr[i].to(dev)
                with torch.autocast("cuda"):
                    p = model(xb)
                    loss = (torch.where(mb, (p - yb) ** 2, torch.zeros_like(p)).sum()
                            / mb.sum()) / accum
                loss.backward(); tot += float(loss)
            opt.step()
            if st % 500 == 0:
                model.eval(); errs = []
                with torch.no_grad():
                    for j in range(0, len(Xva), 2):
                        pv = model(Xva[j:j+2].to(dev)).cpu()
                        errs.append(masked_rmse(pv, Yva[j:j+2], Mva[j:j+2]))
                v = float(np.mean(errs)); model.train()
                print(f"[{name}] {st} loss {tot:.4f} val_rmse {v:.3f}K", flush=True)
                if v < best[0]: best = (v, {k: c.cpu().clone() for k, c in model.state_dict().items()})
        model.load_state_dict(best[1]); model.eval()
        torch.save(best[1], f"{CKPT_DIR}/wv2_{name}.pt")
        return model

    # ---- zero-training baselines on test
    rows_p, rows_f = {L: [] for L in LEADS}, {L: [] for L in LEADS}
    for day, tc, xin, yout in te:
        t0 = bump(tc, -10)
        fl = None
        if t0 in slots:
            fl, _ = estimate_flow(slots[t0][:CROP_H, :CROP_W], slots[tc][:CROP_H, :CROP_W])
        for j, L in enumerate(LEADS):
            y = yout[j]; m = np.isfinite(y) & np.isfinite(xin[-1])
            if m.sum() < 2000: continue
            rows_p[L].append((day, float(np.sqrt(np.nanmean((xin[-1][m] - y[m]) ** 2)))))
            if fl is not None:
                w = pc_warp(xin[-1], fl, L); mm = m & np.isfinite(w)
                rows_f[L].append((day, float(np.sqrt(np.nanmean((w[mm] - y[mm]) ** 2)))))

    results = {"splits": {"train": len(tr), "val": len(va), "test": len(te)},
               "leads_min": [L * 10 for L in LEADS], "ladder": []}
    results["ladder"].append({"model": "persistence",
        "per_lead_K": {L*10: round(agg(rows_p[L]), 4) for L in LEADS if rows_p[L]}})
    results["ladder"].append({"model": "pcflow",
        "per_lead_K": {L*10: round(agg(rows_f[L]), 4) for L in LEADS if rows_f[L]}})
    json.dump(results, open(a.out, "w"), indent=2)
    print("baselines done", flush=True)

    # ---- models (kill bar: vs persistence, full 60-180 aggregate)
    for name, temp, cin, bs, accum in (("unet_spatial", False, 2 * T_IN, a.batch, 1),
                                       ("unet_bissm", True, 2, 2, 4)):
        torch.cuda.empty_cache()
        ckpt = f"{CKPT_DIR}/wv2_{name}.pt"
        if a.eval_only and os.path.exists(ckpt):
            model = UNet(cin, K, temporal=temp).to(dev)
            model.load_state_dict(torch.load(ckpt)); model.eval()
        else:
            model = run_model(name, temp, cin, bs, accum)
        rows = {L: [] for L in LEADS}
        with torch.no_grad():
            for j in range(0, len(te), 2):
                xb = torch.tensor(np.stack([prep_x(w) for w in te[j:j+2]])).to(dev)
                p = model(xb).cpu().numpy() * sd + mu
                for k, (day, tc, xin, yout) in enumerate(te[j:j+2]):
                    for jL, L in enumerate(LEADS):
                        m = np.isfinite(yout[jL])
                        if m.sum() < 2000: continue
                        rows[L].append((day, float(np.sqrt(np.nanmean((p[k, jL][m] - yout[jL][m])**2)))))
        aggp = [r for L in LEADS for r in rows_p[L]]
        aggm = [r for L in LEADS for r in rows[L]]
        se = day_boot(aggp, aggm)
        entry = {"model": name,
                 "per_lead_K": {L*10: round(agg(rows[L]), 4) for L in LEADS if rows[L]},
                 "agg60_180_K": {"model": round(agg(aggm), 4), "persistence": round(agg(aggp), 4),
                                 "gain": round(agg(aggp) - agg(aggm), 4), "se": round(se, 4),
                                 "pass_2se": bool(agg(aggp) - agg(aggm) >= 2 * se)}}
        results["ladder"] = [e for e in results["ladder"] if e["model"] != name] + [entry]
        print(json.dumps(entry), flush=True)
        json.dump(results, open(a.out, "w"), indent=2)
    print("PILOT WV2 DONE ->", a.out, flush=True)

if __name__ == "__main__":
    main()
