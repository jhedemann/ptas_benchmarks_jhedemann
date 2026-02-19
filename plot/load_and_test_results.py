# %% IMPORTS

import pickle
import matplotlib.pyplot as plt

from utils.load_intracranial_data import load_sw_annotation

# %% LOAD DATA

with open("results/run_all/run32/results_twave_all_p17_c2.pkl", "rb") as f:
    results_twave = pickle.load(f)

# %% SHOW MASTER PLOT OF RESULTS WITH GROUND TRUTH ANNOTATION

results_twave.plot_timeseries(ground_truth_sw="data/annotated/Patient17_Channel2_negSWs.npy",
                              ground_truth_ied="data/annotated/Patient17_Channel2_IEDs.npy")
plt.show()
