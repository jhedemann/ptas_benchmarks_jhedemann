# %% IMPORTS

import numpy as np
from pathlib import Path
import re
import pickle
from scipy.signal import butter, sosfiltfilt, resample_poly
import matplotlib.pyplot as plt

from load_intracranial_data import load_data_as_dataset, load_sw_annotation
from analyze_time_frequency import get_signal_subsets_from_events

import Simulations
from Algo_ZeroCrossing import PhaseTracker as ZeroCross


# %%

time_excerpt = 0 # seconds
sampling_rate = 512 # hz

# %% RUN ALGO ON ALL PARTICIPANTS AND CHANNELS

DATA_DIR = Path("data/annotated")

pat = re.compile(r"^Patient(?P<p>\d+)_Channel(?P<c>\d+)_(?P<kind>.+)\.npy$")

def parse_name(fname: str):
    m = pat.match(fname)
    if not m:
        return None
    return int(m["p"]), int(m["c"]), m["kind"]

# 1) Index all files by (patient, channel, kind)
index = {}
for fp in DATA_DIR.glob("*.npy"):
    parsed = parse_name(fp.name)
    if not parsed:
        continue
    p, c, kind = parsed
    index[(p, c, kind)] = fp

# 2) Collect all EEG pairs we can run
pairs = []
for (p, c, kind), eeg_fp in index.items():
    if kind != "EEG":
        continue
    negsw_fp = index.get((p, c, "negSWs"))  # only negative slow waves
    pairs.append((p, c, eeg_fp, negsw_fp))

pairs.sort()

# %%

for p, c, eeg_fp, negsw_fp in pairs:

    if p <= 4:
        continue

    ds = load_data_as_dataset(npy_path=eeg_fp, fs=sampling_rate)

    result = Simulations.run_simulations(ds, ZeroCross(fs=ds.fs, amp_q=0.05))

    with open(f"results/run_all/run19/results_zerocrossrun_all_p{p}_c{c}.pkl", "wb") as f:
        pickle.dump(result, f)
