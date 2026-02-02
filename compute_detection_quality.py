# %% IMPORTS

import pickle
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from Algo_ZeroCrossing import RollingQuantileThreshold
from Algo_Laurent import _OnlineSOSFilter
import Simulations

# %% PREP

# Parse simulation results file
all_runs = os.listdir("results/run_all")
print(all_runs[1])
with open(f"results/run_all/{all_runs[1]}", "rb") as f:
    results_twave = pickle.load(f)

# Parse ground_truth npy files
path_sw = Path("data/annotated/Patient02_Channel2_negSWs.npy")
path_ied = Path("data/annotated/Patient02_Channel2_IEDs.npy")

arr_sw = np.load(path_sw)
arr_ied = np.load(path_ied)

arr_sw_trunc = np.array([x for x in arr_sw if x <= results_twave.Dataset.t.max()])
arr_ied_trunc = np.array([x for x in arr_ied if x <= results_twave.Dataset.t.max()])

print(len(arr_sw_trunc))

sim_stim_times = np.array(results_twave.stims_sp) / results_twave.Dataset.fs

#print(arr_sw_trunc)
#print(arr_ied_trunc)

# %% FUNCTIONS

def get_p_c_struct(data_path):

    all_paths = os.listdir(data_path)
    all_paths.sort()
    eeg_paths = [path for path in all_paths if path[-7:] == "EEG.npy"]

    struct_dict = defaultdict(list)

    for path in eeg_paths:

        patient = path[7:9]
        channel = path[17]
        struct_dict[patient].append(channel)

    return struct_dict

def compute_detection_quality(detected_sim, detected_true, tol=0.1):
    detected_sim = np.array(detected_sim)
    detected_true = np.array(detected_true)

    used_true = np.zeros(len(detected_true), dtype=bool)

    TP = 0
    FP = 0

    for d in detected_sim:
        # compute distances to every unused ground truth event
        diffs = np.abs(detected_true - d)

        # find closest ground truth slow wave
        i = np.argmin(diffs)
        if diffs[i] <= tol and not used_true[i]:
            TP += 1
            used_true[i] = True
        else:
            FP += 1

    FN = int(np.sum(~used_true))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1 = 2 * TP / (2 * TP + FP + FN) if (2*TP + FP + FN) > 0 else np.nan

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "f1": float(f1)
    }

# %% MAIN SCRIPT

tol_qualities = []
tol_range = range(10)

for tol in tol_range:
    tol_qualities.append(compute_detection_quality(sim_stim_times, arr_sw_trunc, tol=tol))

sens, prec, f1 = ([q["sensitivity"] for q in tol_qualities],
                  [q["precision"] for q in tol_qualities],
                  [q["f1"] for q in tol_qualities])

plt.figure(figsize=(8, 5))

plt.plot(tol_range, sens, label="sensitivity (true detected / all slow waves)")
plt.plot(tol_range, prec, label="precision (true detected / all detected)")
plt.plot(tol_range, f1, label="f1 score")

plt.xlabel("tolerance (s)")
plt.ylabel("score")
#plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% COMPUTE STATISTICS ACROSS ALL PATIENTS

# Regex to extract p and c from: results_zerocross_run_all_p3_c1.pkl
result_pat = re.compile(r"p(?P<p>\d+)_c(?P<c>\d+)")
run_num = 29

all_results = []
tol_range = np.linspace(0.1, 1, num=10)

for res_file in os.listdir(f"results/run_all/run{run_num}"):
    # extract IDs
    match = result_pat.search(res_file)
    if not match: continue
    p, c = match.group('p'), match.group('c')
    
    # construct the specific ground truth filename
    # assuming the format is Patient03_Channel1_negSWs.npy
    gt_filename = f"Patient{int(p):02d}_Channel{c}_negSWs.npy"
    gt_path = os.path.join("data/annotated", gt_filename)
    
    if int(p) >= 12:
        continue

    if not os.path.exists(gt_path):
        print(f"Warning: No ground truth found for P{p} C{c}")
        continue

    # load and process
    with open(f"results/run_all/run{run_num}/{res_file}", "rb") as f:
        this_result = pickle.load(f)
    
    this_times = np.array(this_result.stims_sp) / this_result.Dataset.fs
    this_sws = np.load(gt_path)
    this_sws = np.array([x for x in this_sws if x <= this_result.Dataset.t.max()])

    # Check for empty ground truth
    if len(this_sws) == 0:
        print(f"Skipping P{p} C{c}: No ground truth slow waves found.")
        
        # append NaNs if you need to keep a fixed index
        tol_results = [{"sensitivity": np.nan, "precision": np.nan, "f1": np.nan} for _ in tol_range]
        all_results.append(tol_results)
        continue

    tol_results = []
    for tol in tol_range:
        this_stats = compute_detection_quality(this_times, this_sws, tol=tol)
        tol_results.append(this_stats)

    all_results.append(tol_results)
all_results = np.array(all_results)

print(all_results.shape)

# %%

print(all_results)
print(len(tol_results))
# %%

data_struct = get_p_c_struct("data/annotated")
print(data_struct)

# %% COMPUTE AND PLOT STATS FOR ALL PATIENTS AND CHANNELS

all_stats = []

for result in all_results:
    sens, prec, f1 = ([q["sensitivity"] for q in result],
                  [q["precision"] for q in result],
                  [q["f1"] for q in result])
    all_stats.append((sens, prec, f1))

all_stats = np.array(all_stats)
mean_all = np.nanmean(all_stats, axis=0)
std_all = np.nanstd(all_stats, axis=0)
print(mean_all)

plt.figure(figsize=(8, 5))

plt.plot(tol_range, mean_all[0], label="sensitivity (true detected / all slow waves)")
#plt.fill_between(tol_range, mean_all[0] - std_all[0], mean_all[0] + std_all[0], color='blue', alpha=0.2, label='$\pm$ 1 SD')

plt.plot(tol_range, mean_all[1], label="precision (true detected / all detected)")
plt.fill_between(tol_range, mean_all[1] - std_all[1], mean_all[1] + std_all[1], color='orange', alpha=0.2, label='$\pm$ 1 SD')

plt.plot(tol_range, mean_all[2], label="f1 score")
#plt.fill_between(tol_range, mean_all[2] - std_all[2], mean_all[2] + std_all[2], color='green', alpha=0.2, label='$\pm$ 1 SD')


plt.xlabel("tolerance (s)")
plt.ylabel("score")
#plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %%

best_run = np.nanargmax(all_stats[:,1,-1])

print(all_stats[:,1,-1])
print(best_run)
print(np.min(all_stats[:,1,-1]))
print(p)

print(os.listdir(f"results/run_all/run{run_num}")[best_run])

# %%
print(all_stats[27,1,:])
# %% PLOT HISTOGRAM OF PRECISION VALUES AT TOLERANCE 1s

precision_at_1s = all_stats[:, 1, -1]

# filter out NaNs
clean_precision = precision_at_1s[~np.isnan(precision_at_1s)]

# plot histogram
plt.figure(figsize=(9, 6))
plt.hist(clean_precision, bins=np.arange(0, 1.1, 0.05), 
         color='skyblue', edgecolor='black', alpha=0.8)

# Add a vertical line for the mean
plt.axvline(np.mean(clean_precision), color='red', linestyle='--', 
            label=f'mean precision: {np.mean(clean_precision):.3f}')

plt.title("distribution of precision across all channels (tol = 1.0s)")
plt.xlabel("precision")
plt.ylabel("count (number of channels)")
plt.xlim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# %%
