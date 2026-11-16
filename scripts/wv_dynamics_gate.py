#!/usr/bin/env python3
"""T3 dynamics gate for GOES water-vapor channel (C08 BT) nowcasting.

No DQF mask (IR works day+night) — the eval mask is finite(a_tgt) &
finite(a_cur) & finite(warp). Flow estimation: block phase correlation
(pcflow; see src/pcflow.py for why cv2's flow stack is not used).

Cadence: the archive holds true 10-min C08 (days 155-175) plus the older
20-min MCMIPF days (152-154). STEP is a CLI arg (--step, default 10); only
exactly-STEP-min-separated pairs are used, so 20-min days simply don't pair
at STEP=10.

Gate PASS = flow beats persistence by >= 2 x day-bootstrap se on the >=30-min
aggregate.

Usage: python3 scripts/wv_dynamics_gate.py [--step 10]
(writes results/wv_gate_T3_step{STEP}.json)
"""
import os, sys, glob, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import WV_DATA, RESULTS, bump
from src.pcflow import estimate_flow, warp as pc_warp

def load_slots():
    files = sorted(glob.glob(f"{WV_DATA}/*.npz"), key=lambda f: f.split("_s")[-1][:13])
    return [(f.split("_s")[-1][:13], np.load(f)["bt"]) for f in files]

def main():
    a = sys.argv
    STEP = int(a[a.index("--step") + 1]) if "--step" in a else 10
    slots = load_slots()
    by_t = {s[0]: s for s in slots}
    times = [s[0] for s in slots]
    leads = list(range(1, 7))                        # x STEP min
    per_lead = {L: [] for L in leads}
    n_cases = {L: 0 for L in leads}
    for i, t in enumerate(times):
        t0 = bump(t, -STEP)
        if t0 not in by_t: continue
        a_src, a_cur = by_t[t0][1], by_t[t][1]
        fl, _ = estimate_flow(a_src, a_cur)          # one STEP-min flow
        for L in leads:
            tt = bump(t, L * STEP)
            if tt not in by_t: continue
            a_tgt = by_t[tt][1]
            warp = pc_warp(a_cur, fl, L)
            m = np.isfinite(a_tgt) & np.isfinite(a_cur) & np.isfinite(warp)
            if m.sum() < 2000: continue
            err_p = (a_cur - a_tgt)[m]
            err_f = (warp - a_tgt)[m]
            per_lead[L].append((t[:7],
                                float(np.sqrt((err_p ** 2).mean())),
                                float(np.sqrt((err_f ** 2).mean())),
                                t))                   # keep slot id for diag
            n_cases[L] += 1
    rng = np.random.default_rng(0)
    report = {"n_cases": n_cases, "leads_min": {L * STEP: {} for L in leads}}
    for L in leads:
        rows = per_lead[L]
        if not rows: continue
        days = sorted(set(r[0] for r in rows))
        byday = {d: [r for r in rows if r[0] == d] for d in days}
        def stats(rows_):
            p = np.sqrt(np.mean([r[1] ** 2 for r in rows_]))
            f = np.sqrt(np.mean([r[2] ** 2 for r in rows_]))
            return p, f
        p, f = stats(rows)
        diffs = []
        for _ in range(1000):
            sel = rng.choice(days, size=len(days), replace=True)
            pool_ = [r for d in sel for r in byday[d]]
            pp, ff = stats(pool_)
            diffs.append(pp - ff)
        se = float(np.std(diffs))
        report["leads_min"][L * STEP] = {
            "rmse_persistence": round(p, 4), "rmse_flow": round(f, 4),
            "gain_abs": round(p - f, 4), "gain_rel_%": round(100 * (p - f) / p, 2),
            "se_day_bootstrap": round(se, 4), "n": len(rows), "n_days": len(days)}
    # aggregate the >=30-min leads (L*STEP >= 30)
    agg30 = [r for L in leads if L * STEP >= 30 for r in per_lead[L]]
    if agg30:
        days = sorted(set(r[0] for r in agg30))
        byday = {d: [r for r in agg30 if r[0] == d] for d in days}
        p = np.sqrt(np.mean([r[1] ** 2 for r in agg30]))
        f = np.sqrt(np.mean([r[2] ** 2 for r in agg30]))
        diffs = []
        for _ in range(1000):
            sel = rng.choice(days, size=len(days), replace=True)
            pool_ = [r for d in sel for r in byday[d]]
            diffs.append(np.sqrt(np.mean([r[1] ** 2 for r in pool_])) -
                         np.sqrt(np.mean([r[2] ** 2 for r in pool_])))
        se = float(np.std(diffs))
        gate_pass = (p - f) > 0 and (p - f) >= 2 * se
        report["gate_30_60min"] = {
            "rmse_persistence": round(p, 4), "rmse_flow": round(f, 4),
            "gain": round(p - f, 4), "se": round(se, 4),
            "gate": "PASS" if gate_pass else "FAIL",
            "threshold_2se": round(2 * se, 4)}
    out = os.path.join(RESULTS, f"wv_gate_T3_step{STEP}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report.get("gate_30_60min", {}), indent=2))
    for L in leads:
        print(L * STEP, "min:", report["leads_min"][L * STEP])
    print("WV GATE DONE ->", out, flush=True)

if __name__ == "__main__":
    main()
