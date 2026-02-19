# %% IMPORTS

import numpy as np
from pathlib import Path
import re
import pickle
from scipy.signal import butter, sosfiltfilt, resample_poly
import matplotlib.pyplot as plt

from utils.load_intracranial_data import load_data_as_dataset, load_sw_annotation
from utils.analyze_time_frequency import get_signal_subsets_from_events

import algos.Simulations
from algos.Algo_AmpTh import PhaseTracker as AmpThreshold

# %% CONFIG

time_excerpt = 0 # seconds
sampling_rate = 512 # hz

DATA_DIR = Path("data/annotated")
pat = re.compile(r"^Patient(?P<p>\d+)_Channel(?P<c>\d+)_(?P<kind>.+)\.npy$")

# %% RUN ALGO ON ALL PARTICIPANTS AND CHANNELS

def parse_name(fname: str):
    m = pat.match(fname)
    if not m:
        return None
    return int(m["p"]), int(m["c"]), m["kind"]

# index all files by (patient, channel, kind)
index = {}
for fp in DATA_DIR.glob("*.npy"):
    parsed = parse_name(fp.name)
    if not parsed:
        continue
    p, c, kind = parsed
    index[(p, c, kind)] = fp

# collect all runnable EEG pairs
pairs = []
for (p, c, kind), eeg_fp in index.items():
    if kind != "EEG":
        continue
    negsw_fp = index.get((p, c, "negSWs"))  # only negative slow waves
    pairs.append((p, c, eeg_fp, negsw_fp))

pairs.sort()

for p, c, eeg_fp, negsw_fp in pairs:

    ds = load_data_as_dataset(npy_path=eeg_fp, fs=sampling_rate)
    
    sos = butter(4, [0.5, 4.0], btype="bandpass", fs=ds.fs, output="sos")
    ds_filtered = sosfiltfilt(sos, ds.signal.squeeze().astype(float))

    ds_sim = Simulations.SimulationDataset(t=np.arange(len(ds_filtered)) / ds.fs,
                                           signal=ds_filtered,
                                           fs=ds.fs,
                                           name=ds.name + f"_p{p}-c{c}_ds{ds.fs}")

    result = Simulations.run_simulations(ds_sim, AmpThreshold())

    with open(f"results/run_all/run12/results_ampthreshold_all_p{p}_c{c}.pkl", "wb") as f:
        pickle.dump(result, f)
