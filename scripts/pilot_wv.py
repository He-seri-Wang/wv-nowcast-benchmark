#!/usr/bin/env python3
"""WV nowcasting pilot ladder v1 (GOES-16 C08 BT, 10-min, BOX4 4 km).

Ladder: persistence / pcflow (validated phase-correlation flow) / UNet(spatial)
/ UNet+BiSSM(temporal). Windows: 6 input frames -> 6 leads (10-60 min).
Split by day: train 155-168, val 169-171, test 172-175 (time-ordered, no leak).
Kill bar (pre-declared): BiSSM must beat pcflow by >= 2 x day-block bootstrap
se on the 30-60-min aggregate.

Usage: python3 scripts/pilot_wv.py [--steps 3000] [--gpu 0]
       [--out results/pilot_wv.json] [--only spatial|bissm] [--eval-only]
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import WV_DATA, CKPT_DIR, RESULTS, TRAIN, VAL, TEST, CROP
from src.eval import agg, day_boot
from src.models import UNet
from src.pcflow import estimate_flow, warp as pc_warp
from src.constants import bump

T_IN, T_OUT = 6, 6
CROP_H, CROP_W = CROP

def load_runs():
    """Consecutive 10-min runs from the 10-min days only (155-175)."""
    files = sorted(glob.glob(f"{WV_DATA}/*.npz"), key=lambda f: f.split("_s")[-1][:13])
    slots = {}
    for f in files:
        sid = f.split("_s")[-1][:13]
        if "2024155" <= sid[:7] <= "2024175":
            slots[sid] = np.load(f)["bt"]
    times = sorted(slots)
    runs, cur = [], [times[0]]
    for t in times[1:]:
        if bump(t, -10) == cur[-1]:
            cur.append(t)
        else:
            runs.append(cur); cur = [t]
    runs.append(cur)
    return slots, [r for r in runs if len(r) >= T_IN + T_OUT]

def build_windows(runs, slots):
    """-> list of (day, t_center, input(6,H,W), target(6,H,W)) as float32."""
    Ws = []
    for r in runs:
        for i in range(len(r) - T_IN - T_OUT + 1):
            win = r[i:i + T_IN + T_OUT]
            day = win[T_IN - 1][:7]
            xin = np.stack([slots[t] for t in win[:T_IN]])[:, :CROP_H, :CROP_W]
            yout = np.stack([slots[t] for t in win[T_IN:]])[:, :CROP_H, :CROP_W]
            Ws.append((day, win[T_IN - 1], xin.astype(np.float32), yout.astype(np.float32)))
    return Ws

def main():
    A = argparse.ArgumentParser()
    A.add_argument("--steps", type=int, default=3000)
    A.add_argument("--batch", type=int, default=8)
    A.add_argument("--gpu", type=int, default=0)
    A.add_argument("--out", default=os.path.join(RESULTS, "pilot_wv.json"))
    A.add_argument("--only", default=None, help="run a single model rung (spatial|bissm) and splice into existing JSON")
    A.add_argument("--eval-only", action="store_true", help="load saved checkpoint instead of training")
    a = A.parse_args()
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"

    slots, runs = load_runs()
    W = build_windows(runs, slots)
    tr = [w for w in W if TRAIN[0] <= w[0] <= TRAIN[1]]
    va = [w for w in W if VAL[0] <= w[0] <= VAL[1]]
    te = [w for w in W if TEST[0] <= w[0] <= TEST[1]]
    print(f"windows: train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)

    # normalization from train targets only
    mu = float(np.nanmean(np.concatenate([w[3].ravel() for w in tr[:200]])))
    sd = float(np.nanstd(np.concatenate([w[3].ravel() for w in tr[:200]])))
    print(f"train stats: mu={mu:.2f}K sd={sd:.2f}K", flush=True)

    def prep_x(w):
        x = (w[2] - mu) / sd
        v = np.isfinite(x).astype(np.float32)
        return np.stack([np.nan_to_num(x), v], 1)          # (T, 2, H, W)
    def prep_y(w):
        y = (w[3] - mu) / sd
        return np.nan_to_num(y, nan=0.0), np.isfinite(y)    # (T,H,W) both

    Xtr = torch.tensor(np.stack([prep_x(w) for w in tr]))
    Ytr = torch.tensor(np.stack([prep_y(w)[0] for w in tr]))
    Mtr = torch.tensor(np.stack([prep_y(w)[1] for w in tr]))
    Xva = torch.tensor(np.stack([prep_x(w) for w in va]))
    Yva = torch.tensor(np.stack([prep_y(w)[0] for w in va]))
    Mva = torch.tensor(np.stack([prep_y(w)[1] for w in va]))

    def masked_rmse(p, y, m):
        d = torch.where(m, (p - y) ** 2, torch.zeros_like(p))
        return float((d.sum() / m.sum()).sqrt() * sd)       # Kelvin

    def run_model(name, temporal, cin=2, batch=None, accum=1):
        torch.manual_seed(0)                                # seeded for reproducibility
        model = UNet(cin, T_OUT, temporal=temporal).to(dev)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        n = len(Xtr); best = (1e9, None)
        bs = batch or a.batch
        for st in range(a.steps):
            opt.zero_grad()
            tot = 0.0
            for _ in range(accum):
                i = torch.randint(0, n, (bs,))
                xb, yb, mb = Xtr[i].to(dev), Ytr[i].to(dev), Mtr[i].to(dev)
                with torch.autocast("cuda"):
                    p = model(xb)
                    loss = (torch.where(mb, (p - yb) ** 2, torch.zeros_like(p)).sum()
                            / mb.sum()) / accum
                loss.backward(); tot += float(loss)
            opt.step(); loss_v = tot
            if st % 500 == 0:
                model.eval(); errs = []
                with torch.no_grad():
                    for j in range(0, len(Xva), 2):
                        pv = model(Xva[j:j+2].to(dev)).cpu()
                        errs.append(masked_rmse(pv, Yva[j:j+2], Mva[j:j+2]))
                v = float(np.mean(errs)); model.train()
                print(f"[{name}] {st} loss {loss_v:.4f} val_rmse {v:.3f}K", flush=True)
                if v < best[0]: best = (v, {k: ck.cpu().clone() for k, ck in model.state_dict().items()})
        model.load_state_dict(best[1]); model.eval()
        torch.save(best[1], f"{CKPT_DIR}/wv_{name}.pt")
        return model

    results = {"splits": {"train": len(tr), "val": len(va), "test": len(te)},
               "train_stats_K": {"mu": mu, "sd": sd}, "ladder": []}

    # ---- baseline 1: persistence (zero training) on test
    per_lead = {L: [] for L in range(1, T_OUT + 1)}          # (day, rmse per window)
    for day, tc, xin, yout in te:
        m = np.isfinite(yout)
        last = xin[-1]
        for L in range(1, T_OUT + 1):
            if m[L-1].sum() < 2000: continue
            per_lead[L].append((day, float(np.sqrt(np.nanmean((last - yout[L-1])**2)))))
    # ---- baseline 2: pcflow on test
    for day, tc, xin, yout in te:
        t0 = bump(tc, -10)
        if t0 in slots and tc in slots:
            fl, _ = estimate_flow(slots[t0][:CROP_H, :CROP_W], slots[tc][:CROP_H, :CROP_W])
        else:
            fl = np.zeros(xin.shape[1:] + (2,), np.float32)
        for L in range(1, T_OUT + 1):
            m = np.isfinite(yout[L-1]) & np.isfinite(xin[-1])
            if m.sum() < 2000: continue
            wp = pc_warp(xin[-1], fl, L)
            mm = m & np.isfinite(wp)
            per_lead_pc = per_lead.setdefault(f"pc{L}", [])
            per_lead_pc.append((day, float(np.sqrt(np.nanmean((wp[mm] - yout[L-1][mm])**2)))))

    pers_rows = {L: per_lead[L] for L in range(1, T_OUT + 1)}
    pc_rows = {L: per_lead.get(f"pc{L}", []) for L in range(1, T_OUT + 1)}
    results["ladder"].append({
        "model": "persistence",
        "per_lead_K": {L * 10: round(agg(r), 4) for L, r in pers_rows.items() if r}})
    results["ladder"].append({
        "model": "pcflow",
        "per_lead_K": {L * 10: round(agg(r), 4) for L, r in pc_rows.items() if r}})
    json.dump(results, open(a.out, "w"), indent=2)
    print("baselines done", flush=True)

    # ---- models
    rungs = (("unet_spatial", False, None, 1), ("unet_bissm", True, 2, 4))
    if a.only:
        rungs = [r for r in rungs if r[0].endswith(a.only)]
        prev = json.load(open(a.out)) if os.path.exists(a.out) else {"ladder": []}
        results = prev
    for name, temp, bs, accum in rungs:
        torch.cuda.empty_cache()
        ckpt = f"{CKPT_DIR}/wv_{name}.pt"
        if a.eval_only and os.path.exists(ckpt):
            model = UNet(2 * T_IN if not temp else 2, T_OUT, temporal=temp).to(dev)
            model.load_state_dict(torch.load(ckpt)); model.eval()
        else:
            model = run_model(name, temp, cin=(2 * T_IN if not temp else 2), batch=bs, accum=accum)
        rows = {L: [] for L in range(1, T_OUT + 1)}
        with torch.no_grad():
            for j in range(0, len(te), 2):
                xb = torch.tensor(np.stack([prep_x(w) for w in te[j:j+2]])).to(dev)
                p = model(xb).cpu().numpy() * sd + mu
                for k, (day, tc, xin, yout) in enumerate(te[j:j+2]):
                    for L in range(1, T_OUT + 1):
                        m = np.isfinite(yout[L-1])
                        if m.sum() < 2000: continue
                        rows[L].append((day, float(np.sqrt(np.nanmean((p[k, L-1][m] - yout[L-1][m])**2)))))
        entry = {"model": name,
                 "per_lead_K": {L*10: round(agg(r), 4) for L, r in rows.items() if r}}
        # significance vs pcflow per lead + 30-60min aggregate
        agg30_p = [r for L in range(3, T_OUT+1) for r in pc_rows[L]]
        agg30_m = [r for L in range(3, T_OUT+1) for r in rows[L]]
        se = day_boot(agg30_m, agg30_p)
        entry["agg30_60_K"] = {"model": round(agg(agg30_m), 4), "pcflow": round(agg(agg30_p), 4),
                               "gain": round(agg(agg30_p) - agg(agg30_m), 4), "se": round(se, 4),
                               "pass_2se": bool(agg(agg30_p) - agg(agg30_m) >= 2 * se)}
        results["ladder"] = [e for e in results["ladder"] if e["model"] != name] + [entry]
        print(json.dumps(entry), flush=True)
        json.dump(results, open(a.out, "w"), indent=2)
    print("PILOT WV DONE ->", a.out, flush=True)

if __name__ == "__main__":
    main()
