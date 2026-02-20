# %% IMPORTS

from pathlib import Path
import re
import pickle

from utils.load_bids_data import load_eeg_as_dataset

import utils.Simulations as Simulations
from Algo_TWave import PhaseTracker as TWave

# %%

subject = "001"
session = "001"
task = "sleep"
channel = 0
time_excerpt = 3600

ds = load_eeg_as_dataset(subject=subject,
                         session=session,
                         task=task,
                         channel=channel,
                         max_duration_s=time_excerpt)

result = Simulations.run_simulations(ds, TWave(fs=ds.fs, is_ieeg=False))

with open(f"results/run_on_ear_eeg/results_twave_ear-eeg_p{subject}_s{session}_c{channel}_t{time_excerpt}.pkl", "wb") as f:
    pickle.dump(result, f)