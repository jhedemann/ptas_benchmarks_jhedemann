# %% IMPORTS

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import utils.Simulations as Simulations
from utils.Simulations import PhaseTrackerStatus
import os
from scipy import stats
from scipy.signal import butter, sosfilt, sosfilt_zi
import re

from utils.analyze_time_frequency import get_p_c_struct, filter_for_competing_events, find_minima, find_maxima
from utils.load_intracranial_data import load_data_as_dataset


# %% CONFIG

patient03_channel1_eeg = np.load("data/annotated/Patient03_Channel1_EEG.npy")
patient03_channel1_sws = np.load("data/annotated/Patient03_Channel1_negSWs.npy")
patient03_channel1_ieds = np.load("data/annotated/Patient03_Channel1_IEDs.npy")

with open("results/results_zerocross_patient03_channel1_08_newbackoff_sp.pkl", "rb") as f:
    results_twave = pickle.load(f)

data_dir = "data/annotated"
p_c_struct = get_p_c_struct(data_dir)
run_dir = "results/run_all/run40"
fs = 512

# Regex to extract p and c from: results_zerocross_run_all_p3_c1.pkl
result_pat = re.compile(r"p(?P<p>\d+)_c(?P<c>\d+)")
run_num = 40

# %% COMPUTE MEAN ABSOLUTE AMPLITUDE AROUND GROUND TRUTH SLOW WAVE VS NON-SLOW-WAVE

def index_mask(slow_wave_idx, signal_length, window=512):
    half = window // 2
    mask = np.zeros(signal_length, dtype=bool)

    for idx in slow_wave_idx:
        start = max(0, idx - half)
        end = min(signal_length, idx + half)
        mask[int(start):int(end)] = True

    return mask

sw_idx = (patient03_channel1_sws*fs).astype(int)
ied_idx = (patient03_channel1_ieds*fs).astype(int)

sw_idx_mask = index_mask(sw_idx, len(patient03_channel1_eeg), window=512)
ied_idx_mask = index_mask(ied_idx, len(patient03_channel1_eeg), window=512)

eeg = patient03_channel1_eeg.squeeze()

both_idx_mask = sw_idx_mask + ied_idx_mask

mean_amp_sw = np.mean(np.abs(eeg[sw_idx_mask]))
mean_amp_ied = np.mean(np.abs(eeg[ied_idx_mask]))
mean_amp_non_sw = np.mean(np.abs(eeg[~both_idx_mask]))

median_amp_sw = np.median(np.abs(eeg[sw_idx_mask]))
median_amp_ied = np.median(np.abs(eeg[ied_idx_mask]))
median_amp_non_sw = np.median(np.abs(eeg[~both_idx_mask]))

print(f"mean_amp_sw: {mean_amp_sw}")
print(f"mean_amp_ied: {mean_amp_ied}")
print(f"mean_amp_non_sw: {mean_amp_non_sw}")

print(f"median_amp_sw: {median_amp_sw}")
print(f"median_amp_ied: {median_amp_ied}")
print(f"median_amp_non_sw: {median_amp_non_sw}")

# %% PLOT DISTRIBUTION OF AMPLITUDES INSIDE SLOW WAVE WINDOW

plt.figure(figsize=(8, 5))

plt.hist(eeg[~sw_idx_mask], bins=100, alpha=0.5, density=True, label="non-slow-wave amplitudes")
plt.hist(eeg[sw_idx_mask], bins=100, alpha=0.5, density=True, label="slow wave amplitudes")
plt.hist(eeg[ied_idx_mask], bins=100, alpha=0.5, density=True, label="ied amplitudes")

plt.axvline(mean_amp_sw, color="orange", label="mean_amp_sw", ymin=0, ymax=1)
plt.axvline(mean_amp_ied, color="green", label="mean_amp_ied", ymin=0, ymax=1)
plt.axvline(mean_amp_non_sw, color="blue", label="mean_amp_non_sw", ymin=0, ymax=1)

plt.xlabel("amplitude (microV)")
plt.ylabel("counts")
plt.xlim(-5000, 5000)
plt.grid(True, alpha=0.3, axis="y")
plt.legend()
plt.tight_layout()
plt.show()

# %% PLOT PHASE ESTIMATE HISTOGRAM AT STIMULATION INDICES

results_twave.plot_phase_hist()

# %% PLOT PHASE ESTIMATE HISTOGRAM AT GROUND TRUTH SLOW WAVE INDICES

sw_idx_trunc = np.array([i for i in sw_idx if i/results_twave.Dataset.fs <= results_twave.Dataset.t.max()])

internals = results_twave.internals_ts

sw_phases = np.array([
    internals[int(i)]["phase"]
    for i in sw_idx_trunc
])

print(sw_phases)

Simulations.plot_phase_hist_array(stim_phases=sw_phases, target_phase=0, title="Phase estimate distribution at ground-truth slow waves")

# %% COMPUTE AMOUNTS OF DIFFERENT PHASETRACKER STATUSES

status = np.array(results_twave.status_ts)

def count(flag):
    return np.sum((status & flag) > 0)

print("NONE:", count(PhaseTrackerStatus.NONE))

print("STIM1:", count(PhaseTrackerStatus.STIM1))
print("STIM2:", count(PhaseTrackerStatus.STIM2))

print("INHIBITED:", count(PhaseTrackerStatus.INHIBITED))
print("INHIBITED_AMP:", count(PhaseTrackerStatus.INHIBITED_AMP))
print("INHIBITED_RATIO:", count(PhaseTrackerStatus.INHIBITED_RATIO))
print("INHIBITED_QUADRATURE:", count(PhaseTrackerStatus.INHIBITED_QUADRATURE))

print("WRONGPHASE:", count(PhaseTrackerStatus.WRONGPHASE))
print("BACKOFF:", count(PhaseTrackerStatus.BACKOFF))
print("BACKOFF_ISI:", count(PhaseTrackerStatus.BACKOFF_ISI))

# %% PRINT PHASETRACKER STATUS AT GROUND TRUTH SLOW WAVE INDICES

print(len(results_twave.status_ts))
print(PhaseTrackerStatus.STIM1)

path_sw = Path("data/annotated/Patient03_Channel1_negSWs.npy")
path_ied = Path("data/annotated/Patient03_Channel1_IEDs.npy")

arr_sw = np.load(path_sw)
arr_ied = np.load(path_ied)

arr_sw_trunc = np.array([x * results_twave.Dataset.fs for x in arr_sw if x <= results_twave.Dataset.t.max()], dtype=int)
arr_ied_trunc = np.array([x * results_twave.Dataset.fs for x in arr_ied if x <= results_twave.Dataset.t.max()], dtype=int)


print(results_twave.status_ts[:50])

for i in arr_sw_trunc:
    print(results_twave.status_ts[int(i)].name)
    
# %% INVESTIGATE hl_ratio

internals = results_twave.internals_ts

hl_ratios = []
for i in range(len(internals)):
    hl_ratios.append(internals[i]["hl_ratio"])
hl_ratios = np.array(hl_ratios)
print(len(hl_ratios))
print(hl_ratios[:20])

print(np.sum(hl_ratios))

# %% ANALYZE MINIMAL AMPLITUDE DISTRIBUTION OF IEDs vs SWs

min_amps_sw = []
min_amps_ied = []


for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        dataset = load_data_as_dataset(signal_filepath, fs=fs)

        signal = dataset.signal

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)
        
        # find minima of sws
        sw_minima = find_minima(signal, filtered_arr_sw, fs=fs, window_size_s=4, low_pass=4)

        # find minima of ieds
        ied_minima = find_minima(signal, filtered_arr_ied, fs=fs, window_size_s=4, low_pass=4)

        # compute amplitudes at minima
        this_min_amps_sw = signal[sw_minima]
        this_min_amps_ied = signal[ied_minima]

        # extend list 
        min_amps_sw.extend(this_min_amps_sw)
        min_amps_ied.extend(this_min_amps_ied)

mean_min_amp_sw = np.mean(min_amps_sw)
mean_min_amp_ied = np.mean(min_amps_ied)

print("Mean amplitude of trough low point:")
print("SW: ", mean_min_amp_sw)
print("IED: ", mean_min_amp_ied)

p_range = [20, 10, 5, 1, 0.1]
sw_percentiles = np.nanpercentile(min_amps_sw, p_range)
ied_percentiles = np.nanpercentile(min_amps_ied, p_range)

# Print the results in a readable format
print(f"{'Percentile':<12} | {'SW':<10} | {'IED':<10}")
print("-" * 35)
for p, sw_val, ied_val in zip(p_range, sw_percentiles, ied_percentiles):
    print(f"{p:<12} | {sw_val:>10.2f} | {ied_val:>10.2f}")

# line break
print()

amp_values = [-5000, -4000, -3000, -2000]

# Convert lists to numpy arrays
sw_array = np.array(min_amps_sw)
ied_array = np.array(min_amps_ied)

# Use the arrays for the masking operation
sw_clean = sw_array[~np.isnan(sw_array)]
ied_clean = ied_array[~np.isnan(ied_array)]

# Print the results in the same structure as before
print(f"{'Amplitude':<12} | {'SW %-tile':<10} | {'IED %-tile':<10}")
print("-" * 38)

for val in amp_values:
    sw_p = stats.percentileofscore(sw_clean, val, kind='rank')
    ied_p = stats.percentileofscore(ied_clean, val, kind='rank')
    print(f"{val:<12} | {sw_p:>10.2f}% | {ied_p:>10.2f}%")

# %% ANALYZE MAXIMAL AMPLITUDE DISTRIBUTION OF IEDs vs SWs

max_amps_sw = []
max_amps_ied = []
max_amps_fp = []

for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        dataset = load_data_as_dataset(signal_filepath, fs=fs)

        result_filename = f"results_zerocrossrun_all_p{int(p)}_c{c}.pkl"
        result_filepath = os.path.join(run_dir, result_filename)

        with open(result_filepath, "rb") as f:
            this_result = pickle.load(f)

        detected_sws = np.array(this_result.stims_sp)

        signal = dataset.signal

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)

        # filter such that only detected slow waves that aren't slow waves remain
        filtered_detected_sws, arr_sws = filter_for_competing_events(detected_sws, arr_sws, buffer_s=2)

        # find maxima of sws
        sw_maxima = find_maxima(signal, filtered_arr_sw, fs=fs, window_size_s=1)

        # find maxima of ieds
        ied_maxima = find_maxima(signal, filtered_arr_ied, fs=fs, window_size_s=1)

        # find maxima of false positives
        fp_maxima = find_maxima(signal, filtered_detected_sws, fs=fs, window_size_s=1)

        # compute amplitudes at maxima
        this_max_amps_sw = signal[sw_maxima]
        this_max_amps_ied = signal[ied_maxima]
        this_max_amps_fp = signal[fp_maxima]

        # extend list 
        max_amps_sw.extend(this_max_amps_sw)
        max_amps_ied.extend(this_max_amps_ied)
        max_amps_fp.extend(this_max_amps_fp)

mean_max_amp_sw = np.mean(max_amps_sw)
mean_max_amp_ied = np.mean(max_amps_ied)
mean_max_amps_fp = np.mean(max_amps_fp)

print("")
print(mean_max_amp_sw, mean_max_amp_ied, mean_max_amps_fp)

plt.hist(max_amps_ied, bins=50, alpha=0.3, density=True, label="IED pos amps")
plt.hist(max_amps_sw, bins=50, alpha=0.3, density=True, label="SW pos amps")
plt.hist(max_amps_fp, bins=50, alpha=0.3, density=True, label="FP pos amps")
plt.legend()
plt.show()

p_range = [20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9]
sw_percentiles = np.nanpercentile(max_amps_sw, p_range)
ied_percentiles = np.nanpercentile(max_amps_ied, p_range)
fp_percentiles = np.nanpercentile(max_amps_fp, p_range)

# Print the results in a readable format
print(f"{'Percentile':<12} | {'SW':<10} | {'IED':<10} | {'FP':<10}")
print("-" * 35)
for p, sw_val, ied_val, fp_val in zip(p_range, sw_percentiles, ied_percentiles, fp_percentiles):
    print(f"{p:<12} | {sw_val:>10.2f} | {ied_val:>10.2f} | {fp_val:>10.2f}")

amp_values = [2000, 3000, 4000, 5000]

# Convert lists to numpy arrays
sw_array = np.array(max_amps_sw)
ied_array = np.array(max_amps_ied)

# Use the arrays for the masking operation
sw_clean = sw_array[~np.isnan(sw_array)]
ied_clean = ied_array[~np.isnan(ied_array)]

# Print the results in the same structure as before
print(f"{'Amplitude':<12} | {'SW %-tile':<10} | {'IED %-tile':<10}")
print("-" * 38)

for val in amp_values:
    sw_p = stats.percentileofscore(sw_clean, val, kind='rank')
    ied_p = stats.percentileofscore(ied_clean, val, kind='rank')
    
    print(f"{val:<12} | {sw_p:>10.2f}% | {ied_p:>10.2f}%")

# %% COMPUTE MIN AMPLITUDES AT EVENTS AND MAX AMPLITUDE BEFORE

min_max_amps_fp = []
min_max_amps_sw = []
min_max_amps_ied = []


for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        dataset = load_data_as_dataset(signal_filepath, fs=fs)

        result_filename = f"results_zerocrossrun_all_p{int(p)}_c{c}.pkl"
        result_filepath = os.path.join(run_dir, result_filename)

        with open(result_filepath, "rb") as f:
            this_result = pickle.load(f)

        detected_sws = np.array(this_result.stims_sp)

        signal = dataset.signal

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)

        # filter such that only detected slow waves that aren't slow waves remain
        filtered_detected_sws, arr_sws = filter_for_competing_events(detected_sws, arr_sws, buffer_s=2)

        # find minima of fps
        fp_minima = find_minima(signal, filtered_detected_sws, fs=fs, window_size_s=1, low_pass=4)
        fp_minima = fp_minima.astype(int)

        segments = [signal[max(0, time - 1024) : time] for time in fp_minima]
        max_amps = [np.max(seg) if seg.size > 0 else 0 for seg in segments]
        min_amps = signal[fp_minima]
        
        min_max_amps_fp.extend(zip(min_amps, max_amps))

        # find minima of sws
        sw_minima = find_minima(signal, filtered_arr_sw, fs=fs, window_size_s=1, low_pass=4)
        sw_minima = sw_minima.astype(int)

        segments = [signal[max(0, time - 1024) : time] for time in sw_minima]
        max_amps = [np.max(seg) if seg.size > 0 else 0 for seg in segments]
        min_amps = signal[sw_minima]
        
        min_max_amps_sw.extend(zip(min_amps, max_amps))

        # find minima of ieds
        ied_minima = find_minima(signal, filtered_arr_ied, fs=fs, window_size_s=1, low_pass=4)
        ied_minima = ied_minima.astype(int)

        segments = [signal[max(0, time - 1024) : time] for time in ied_minima]
        max_amps = [np.max(seg) if seg.size > 0 else 0 for seg in segments]
        min_amps = signal[ied_minima]
        
        min_max_amps_ied.extend(zip(min_amps, max_amps))

min_max_amps_fp = np.asarray(min_max_amps_fp)
min_max_amps_sw = np.asarray(min_max_amps_sw)
min_max_amps_ied = np.asarray(min_max_amps_ied)
amp_diffs_fp = np.array([(pair[0], pair[1], pair[1]-pair[0]) for pair in min_max_amps_fp])
amp_diffs_sw = np.array([(pair[0], pair[1], pair[1]-pair[0]) for pair in min_max_amps_sw])
amp_diffs_ied = np.array([(pair[0], pair[1], pair[1]-pair[0]) for pair in min_max_amps_ied])

print("FP mean amp diff ", np.mean(amp_diffs_fp[:,2]))
print("SW mean amp diff ", np.mean(amp_diffs_sw[:,2]))
print("IED mean amp diff ", np.mean(amp_diffs_ied[:,2]))


plt.scatter(min_max_amps_sw[:,1], min_max_amps_sw[:,0])
plt.show()

print(np.corrcoef(min_max_amps_sw[:,1], min_max_amps_sw[:,0]))
print(np.corrcoef(min_max_amps_ied[:,1], min_max_amps_ied[:,0]))

ratio_min_max_fp = np.array([abs(pair[0])/pair[1] for pair in min_max_amps_fp])
# FP Statistics
print("FP statistics")
print(f"Cleaned Mean: {np.mean(ratio_min_max_fp):.3f}")
print(f"Cleaned STD: {np.std(ratio_min_max_fp):.3f}")
print(f"fp Median: {np.median(ratio_min_max_fp):.3f}")
print(f"fp IQR: {np.percentile(ratio_min_max_fp, 75) - np.percentile(ratio_min_max_fp, 25):.3f}")
print()


ratio_min_max_sw = np.array([abs(pair[0])/pair[1] for pair in min_max_amps_sw])
clean_ratios_sw = ratio_min_max_sw[ratio_min_max_sw < 20]
# SW statistics
print("SW statistics")
print(f"Cleaned Mean: {np.mean(clean_ratios_sw):.3f}")
print(f"Cleaned STD: {np.std(clean_ratios_sw):.3f}")
print(f"SW Median: {np.median(ratio_min_max_sw):.3f}")
print(f"SW IQR: {np.percentile(ratio_min_max_sw, 75) - np.percentile(ratio_min_max_sw, 25):.3f}")
print()

ratio_min_max_ied = np.array([abs(pair[0])/pair[1] for pair in min_max_amps_ied])
# IED Statistics
print("IED statistics")
print(f"Cleaned Mean: {np.mean(ratio_min_max_ied):.3f}")
print(f"Cleaned STD: {np.std(ratio_min_max_ied):.3f}")
print(f"IED Median: {np.median(ratio_min_max_ied):.3f}")
print(f"IED IQR: {np.percentile(ratio_min_max_ied, 75) - np.percentile(ratio_min_max_ied, 25):.3f}")
print()

plt.hist(np.sort(ratio_min_max_sw)[10:-10], bins=50, alpha=0.3, density=True, label="SW peak amp ratio")
plt.hist(np.sort(ratio_min_max_ied)[10:-10], bins=50, alpha=0.3, density=True, label="IED peak amp ratio")
plt.legend()
plt.show()

# %% COMPUTE FULL WIDTH AT HALF MAXIMUM 

def get_peak_fwhm(signal, peak_idx, fs=fs, search_win_s=0.2):
    """Calculates the width of a positive peak at 50% of its height."""
    peak_val = signal[peak_idx]
    
    # Define a small window around this specific peak to find its base
    win_samples = int(search_win_s * fs)
    start = max(0, peak_idx - win_samples)
    end = min(len(signal), peak_idx + win_samples)
    region = signal[start:end]
    
    # Relative height: we assume the 'base' is 0 for simplicity, 
    # or you can use np.percentile(region, 5) as a baseline.
    baseline = np.median(region) 
    half_height = baseline + (peak_val - baseline) / 2
    
    # How many samples are above the half-height mark?
    is_above_half = region > half_height
    width_ms = (np.sum(is_above_half) / fs) * 1000
    return width_ms

def get_trough_fwhm(signal, trough_idx, fs=fs, search_win_s=0.2):
    """Calculates the width of a negative trough at 50% of its height."""
    trough_val = signal[trough_idx]
    
    win_samples = int(search_win_s * fs)
    start = max(0, trough_idx - win_samples)
    end = min(len(signal), trough_idx + win_samples)
    region = signal[start:end]
    
    # Baseline is the 'ceiling' (max) of this local area
    baseline = np.max(region) 
    # Half-way point between the ceiling and the floor
    half_depth = baseline - (baseline - trough_val) / 2
    
    # Samples below the half-way line
    is_below_half = region < half_depth
    return (np.sum(is_below_half) / fs) * 1000

fwhm_ratios_fp = []
fwhm_ratios_sws = []
fwhm_ratios_ieds = []

for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        result_filename = f"results_zerocrossrun_all_p{int(p)}_c{c}.pkl"
        result_filepath = os.path.join(run_dir, result_filename)

        with open(result_filepath, "rb") as f:
            this_result = pickle.load(f)

        detected_sws = np.array(this_result.stims_sp)

        dataset = load_data_as_dataset(signal_filepath, fs=fs)

        signal = dataset.signal

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)
        
        # filter such that only detected slow waves that aren't slow waves remain
        filtered_detected_sws, arr_sws = filter_for_competing_events(detected_sws, arr_sws, buffer_s=2)

        # find minima of fp detections
        fp_minima = find_minima(signal, filtered_detected_sws, fs=fs, window_size_s=1, low_pass=4)
        fp_minima = fp_minima.astype(int)

        # find minima of sws
        sw_minima = find_minima(signal, filtered_arr_sw, fs=fs, window_size_s=1, low_pass=4)
        sw_minima = sw_minima.astype(int)

        # find minima of sws
        ied_minima = find_minima(signal, filtered_arr_ied, fs=fs, window_size_s=1, low_pass=4)
        ied_minima = ied_minima.astype(int)

        for trough_idx in fp_minima:

            t_width = get_trough_fwhm(signal, trough_idx)

            # look back 500ms (256 samples) to find the 'uptick'
            lookback = 256
            search_start = max(0, trough_idx - lookback)
            
            # find the local maximum (the peak) before the trough
            pre_window = signal[search_start : trough_idx]
            if pre_window.size == 0:
                fwhm_ratios_ieds.append(np.nan)
                continue
                
            # index of the max relative to the search_start
            local_peak_idx = np.argmax(pre_window)
            absolute_peak_idx = search_start + local_peak_idx
            
            # compute the width of that specific peak
            p_width = get_peak_fwhm(signal, absolute_peak_idx, fs=fs)

            if t_width > 0:
                fwhm_ratios_fp.append(p_width/t_width)

        for trough_idx in sw_minima:

            t_width = get_trough_fwhm(signal, trough_idx)

            # look back 500ms (256 samples) to find the 'uptick'
            lookback = 256
            search_start = max(0, trough_idx - lookback)
            
            # find the local maximum (the peak) before the trough
            pre_window = signal[search_start : trough_idx]
            if pre_window.size == 0:
                fwhm_ratios_ieds.append(np.nan)
                continue
                
            # index of the max relative to the search_start
            local_peak_idx = np.argmax(pre_window)
            absolute_peak_idx = search_start + local_peak_idx
            
            # compute the width of that specific peak
            p_width = get_peak_fwhm(signal, absolute_peak_idx, fs=fs)

            if t_width > 0:
                fwhm_ratios_sws.append(p_width/t_width)

        for trough_idx in ied_minima:

            t_width = get_trough_fwhm(signal, trough_idx)

            # look back 500ms (256 samples) to find the 'uptick'
            lookback = 256
            search_start = max(0, trough_idx - lookback)
            
            # find the local maximum (the peak) before the trough
            pre_window = signal[search_start : trough_idx]
            if pre_window.size == 0:
                fwhm_ratios_ieds.append(np.nan)
                continue
                
            # index of the max relative to the search_start
            local_peak_idx = np.argmax(pre_window)
            absolute_peak_idx = search_start + local_peak_idx
            
            # compute the width of that specific peak
            p_width = get_peak_fwhm(signal, absolute_peak_idx, fs=fs)

            if t_width > 0:
                fwhm_ratios_ieds.append(p_width/t_width)

fwhm_ratios_fp = np.array(fwhm_ratios_fp)
fwhm_ratios_sws = np.array(fwhm_ratios_sws)
fwhm_ratios_ieds = np.array(fwhm_ratios_ieds)

# filter out extreme ratios (e.g., > 2.0) to see a cleaner mean/std
clean_ied_ratios = fwhm_ratios_ieds[fwhm_ratios_ieds < 2.0]

print("FP stats:")
print(f"Mean: {np.mean(fwhm_ratios_fp):.3f}")
print(f"STD: {np.std(fwhm_ratios_fp):.3f}")
print(f"fp Median: {np.median(fwhm_ratios_fp):.3f}")
print(f"fp IQR: {np.percentile(fwhm_ratios_fp, 75) - np.percentile(fwhm_ratios_fp, 25):.3f}")

print("SW stats:")
print(f"Mean: {np.mean(fwhm_ratios_sws):.3f}")
print(f"STD: {np.std(fwhm_ratios_sws):.3f}")
print(f"SW Median: {np.median(fwhm_ratios_sws):.3f}")
print(f"SW IQR: {np.percentile(fwhm_ratios_sws, 75) - np.percentile(fwhm_ratios_sws, 25):.3f}")

print("IED stats:")
print(f"Cleaned IED Mean: {np.mean(clean_ied_ratios):.3f}")
print(f"Cleaned IED STD: {np.std(clean_ied_ratios):.3f}")
print(f"IED Median: {np.median(fwhm_ratios_ieds):.3f}")
print(f"IED IQR: {np.percentile(fwhm_ratios_ieds, 75) - np.percentile(fwhm_ratios_ieds, 25):.3f}")

plt.figure(figsize=(10, 6))

# plotting the distributions
plt.hist(fwhm_ratios_sws, bins=50, alpha=0.5, label='Slow Waves', color='blue', range=(0, 1.5))
plt.hist(fwhm_ratios_ieds, bins=50, alpha=0.5, label='IEDs', color='orange', range=(0, 1.5))

# add lines for the medians
plt.axvline(np.median(fwhm_ratios_sws), color='blue', linestyle='dashed', linewidth=2, label='SW Median')
plt.axvline(np.median(fwhm_ratios_ieds), color='orange', linestyle='dashed', linewidth=2, label='IED Median')

plt.title('Distribution of Sharpness Ratios (Peak Width / Trough Width)')
plt.xlabel('Sharpness Ratio (Lower = Sharper Pre-Peak)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(fwhm_ratios_sws.shape)
print(fwhm_ratios_ieds.shape)

# %% COMPUTE HIGH FREQUENCY AMPLITUDE ENERGY AROUND EVENTS 

max_amps_sw = []
max_amps_ied = []
max_amps_fp = []
max_amps_fn = []

for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        dataset = load_data_as_dataset(signal_filepath, fs=fs)

        result_filename = f"results_zerocrossrun_all_p{int(p)}_c{c}.pkl"
        result_filepath = os.path.join(run_dir, result_filename)

        with open(result_filepath, "rb") as f:
            this_result = pickle.load(f)

        detected_sws = np.array(this_result.stims_sp)

        signal = dataset.signal

        sos = butter(4, 90, btype="highpass", fs=fs, output="sos")
        zi = sosfilt_zi(sos)
        hf_signal, zi = sosfilt(sos, signal, zi=zi)

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)
        
        # filter such that only detected slow waves that aren't slow waves remain
        filtered_detected_sws, filtered_undetected_sws = filter_for_competing_events(detected_sws, arr_sws, buffer_s=2)

        # find maxima of sws
        sw_minima = find_minima(signal, filtered_arr_sw, fs=fs, window_size_s=1)
        sw_maxima = find_maxima(hf_signal, sw_minima/fs-0.15, fs=fs, window_size_s=0.2)
        # find maxima of ieds
        ied_minima = find_minima(signal, filtered_arr_ied, fs=fs, window_size_s=1)
        ied_maxima = find_maxima(hf_signal, ied_minima/fs-0.15, fs=fs, window_size_s=0.2)
        # find maxima of false positives
        fp_minima = find_minima(signal, filtered_detected_sws, fs=fs, window_size_s=1)
        fp_maxima = find_maxima(hf_signal, fp_minima/fs-0.15, fs=fs, window_size_s=0.2)
        # find maxima of false negatives
        fn_minima = find_minima(signal, filtered_undetected_sws, fs=fs, window_size_s=1)
        fn_maxima = find_maxima(hf_signal, fn_minima/fs-0.15, fs=fs, window_size_s=0.2)

        # compute high frequency amplitude 150 ms before minimum
        this_max_amps_sw = hf_signal[sw_maxima]
        this_max_amps_ied = hf_signal[ied_maxima]
        this_max_amps_fp = hf_signal[fp_maxima]
        this_max_amps_fn = hf_signal[fn_maxima]

        # extend list 
        max_amps_sw.extend(this_max_amps_sw)
        max_amps_ied.extend(this_max_amps_ied)
        max_amps_fp.extend(this_max_amps_fp)
        max_amps_fn.extend(this_max_amps_fn)

print(len(max_amps_sw))
print(len(max_amps_ied))
print(len(max_amps_fp))
print(len(max_amps_fn))

mean_max_amp_sw = np.mean(max_amps_sw)
mean_max_amp_ied = np.mean(max_amps_ied)
mean_max_amp_fp = np.mean(max_amps_fp)
mean_max_amp_fn = np.mean(max_amps_fn)

print("")
print(mean_max_amp_sw, mean_max_amp_ied, mean_max_amp_fp, mean_max_amp_fn)

plt.hist(max_amps_fp, bins=20, alpha=0.3, density=True, label="FP pos amps")
plt.hist(max_amps_fn, bins=20, alpha=0.3, density=True, label="FN pos amps")

plt.legend()
plt.show()

p_range = [20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9]
sw_percentiles = np.nanpercentile(max_amps_sw, p_range)
ied_percentiles = np.nanpercentile(max_amps_ied, p_range)
fp_percentiles = np.nanpercentile(max_amps_fp, p_range)
fn_percentiles = np.nanpercentile(max_amps_fn, p_range)


# Print the results in a readable format
print(f"{'Percentile':<12} | {'SW':<10} | {'IED':<10} | {'FP':<10} | {'FN':<10}")
print("-" * 35)
for p, sw_val, ied_val, fp_val, fn_val in zip(p_range, sw_percentiles, ied_percentiles, fp_percentiles, fn_percentiles):
    print(f"{p:<12} | {sw_val:>10.2f} | {ied_val:>10.2f} | {fp_val:>10.2f} | {fn_val:>10.2f}") 

# %% INSPECT QUADRATURE VALUES

all_internals = []
stim_internals = []
first_internals = []

all_statuses = []

stim_internals_dict = {}

for res_file in os.listdir(f"results/run_all/run{run_num}"):
    # extract IDs
    match = result_pat.search(res_file)
    if not match: continue
    p, c = match.group('p'), match.group('c')
    
    # construct the specific ground truth filename
    # assuming the format is Patient03_Channel1_negSWs.npy
    gt_filename = f"Patient{int(p):02d}_Channel{c}_negSWs.npy"
    gt_path = os.path.join("data/annotated", gt_filename)
    
    if not os.path.exists(gt_path):
        print(f"Warning: No ground truth found for P{p} C{c}")
        continue

    # load and process
    with open(f"results/run_all/run{run_num}/{res_file}", "rb") as f:
        this_result = pickle.load(f)
    
    this_internals = np.array(this_result.internals_ts) # / this_result.Dataset.fs
    this_times = np.array(this_result.stims_sp, dtype=int)
    this_first = np.array(this_result.wave_detected_sp, dtype=int)
    this_statuses = np.array(this_result.status_ts)

    this_stim_internals = this_internals[this_times]
    this_first_internals = this_internals[this_first]

    all_internals.extend(this_internals)
    stim_internals.extend(this_stim_internals)
    first_internals.extend(this_first_internals)
    all_statuses.extend(this_statuses)

    stim_internals_dict[f"{p}_{c}"] = this_stim_internals

all_internals = np.array(all_internals)
stim_internals = np.array(stim_internals)
first_internals = np.array(first_internals)
all_statuses = np.array(all_statuses)

print(all_internals.shape)
print(stim_internals.shape)
print(first_internals.shape)

all_quadratures = np.array([i["quadrature"] for i in all_internals])
stim_quadratures = np.array([i["quadrature"] for i in stim_internals])
first_quadratures = np.array([i["quadrature"] for i in first_internals])

print(np.mean(all_quadratures))
print(np.mean(stim_quadratures))
print(np.mean(first_quadratures))

plt.hist(stim_quadratures, bins=50, alpha=0.5, density=True, label="stim quadrature values")
plt.hist(first_quadratures, bins=50, alpha=0.5, density=True, label="first quadrature values")

plt.legend()
plt.show()

all_phases = np.array([i["phase"] for i in all_internals])
stim_phases = np.array([i["phase"] for i in stim_internals])
first_phases = np.array([i["phase"] for i in first_internals])

print(np.mean(all_phases))
print(np.mean(stim_phases))
print(np.mean(first_phases))

plt.hist(stim_phases, bins=50, alpha=0.5, density=True, label="stim phase values")
plt.hist(first_phases, bins=50, alpha=0.5, density=True, label="first phase values")

plt.legend()
plt.show()

wrongphase_idx = np.array(np.where(all_statuses==32))

print(type(wrongphase_idx))
wrongphase_quad = all_quadratures[wrongphase_idx]
non_wrongphase_quad = all_quadratures[~wrongphase_idx]

plt.hist(wrongphase_quad, bins=50, alpha=0.5, density=True, label="wrongphase quadrature values")

plt.legend()
plt.show()
print(stim_internals_dict)

all_amps = np.array([i["amp"] for i in all_internals[:60*512]])

plt.plot(all_amps)
