"""Central constants for the WV nowcast benchmark (single source of truth).

Paths are overridable via environment variables so the repo runs anywhere;
defaults point at the original experiment locations.
"""
import os
import datetime as dt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- data locations (override via env) ---------------------------------------
WV_DATA = os.environ.get("WV_DATA", "/scratch/zceehw4/earthmm/wv_pre")      # decoded 4 km frames
GOES_RAW = os.environ.get("WV_RAW", "/scratch/zceehw4/earthmm/goes16")      # ABI-L2 MCMIPF .nc
C08_RAW = os.environ.get("WV_C08_RAW", "/scratch/zceehw4/earthmm/c08_raw")  # L1b download scratch
CKPT_DIR = os.environ.get("WV_CKPT", "/scratch/zceehw4/earthmm")            # trained weights
RESULTS = os.path.join(REPO, "results")

# --- domain --------------------------------------------------------------------
# GOES-16 box on the 4 km pooled grid. Native 2 km crops are 2x these bounds
# with PAD rows/cols of slack (see scripts/fetch_c08.py::decode).
BOX4 = (490, 1015, 2146, 2635)   # (r0, r1, c0, c1) on the 4 km grid
PAD = 6                          # native 2 km padding around the 2x-BOX4 crop
CROP = (530, 494)                # (H, W) crop applied to pooled frames downstream

# --- experiment split (YYYYDDD day strings, inclusive) --------------------------
TRAIN = ("2024155", "2024168")
VAL = ("2024169", "2024171")
TEST = ("2024172", "2024175")
ARCHIVE_DAYS = ("2024155", "2024175")   # decoded archive span


def bump(t, dmin):
    """Shift a YYYYDDDHHMMSS slot id by dmin minutes."""
    b = dt.datetime.strptime(t, "%Y%j%H%M%S")
    return (b + dt.timedelta(minutes=dmin)).strftime("%Y%j%H%M%S")
