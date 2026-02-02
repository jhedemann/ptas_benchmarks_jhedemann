# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from load_intracranial_data import load_sw_annotation

from Simulations import PhaseTrackerStatus

# %% LOAD DATA

with open("results/run_all/run28/results_laurent_all_p16_c6.pkl", "rb") as f:
    results_twave = pickle.load(f)


print("len(t):", len(results_twave.Dataset.t))
print("len(signal):", len(results_twave.Dataset.signal))
print("len(status_ts):", len(results_twave.status_ts))
print("len(internals_ts):", len(results_twave.internals_ts))
print("block_size_sp in tracker?:", getattr(results_twave, "block_size_sp", None))


# %%
print(results_twave.PhaseTracker.__dict__)
# %% SHOW MASTER PLOT OF RESULTS WITH GROUND TRUTH ANNOTATION

results_twave.plot_timeseries(ground_truth_sw="data/annotated/Patient16_Channel6_negSWs.npy",
                              ground_truth_ied="data/annotated/Patient16_Channel6_IEDs.npy")
plt.show()


#print(results_twave.Dataset.t.shape)
print(len(results_twave.stims_sp))
print(np.diff(results_twave.Dataset.t)*results_twave.Dataset.fs)

print(results_twave.Dataset.t.max())

# %%

neg_sw_annot = "/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient02_Channel2_negSWs.npy"
arr_neg_sw = load_sw_annotation(neg_sw_annot)
print(len(arr_neg_sw))


# %%

# %%

#with open("results/run_all/run02/results_zerocross_run_all_p2_c2.pkl", "rb") as f:
#    results_1 = pickle.load(f)

with open("results/run_all/run03/results_zerocross_run_all_p2_c2.pkl", "rb") as f:
    results_2 = pickle.load(f)

#print(len(results_1.stims_sp))
print(len(results_2.stims_sp))

# %%
