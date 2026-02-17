# %% IMPORTS

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from math import pi

from analyze_time_frequency import find_minima, get_signal_subsets_from_events, get_p_c_struct

# %% CONFIG


# %% FUNCTIONS

def get_stim_waveforms(eeg, stims):
    """
    Takes an EEG trace and an array of indices and
    returns subsets of the signal aligned to the trough around each index.
    """
    stims_minima = find_minima(eeg, stims, window_size_s=2)
    stims_subsets = get_signal_subsets_from_events(eeg, stims_minima, window_size_s=1)

    return stims_subsets

def plot_avg_waveform(ax,
                      waveforms):
    waveforms = np.array(waveforms)
    print(waveforms.shape)
    avg_waveform = np.mean(waveforms, axis=0)
    n = waveforms.shape[0]
    sem_waveform = np.std(waveforms, axis=0) / np.sqrt(n)
    time_axis = np.linspace(-2, 2, avg_waveform.shape[0])

    ax.fill_between(time_axis, avg_waveform - sem_waveform, avg_waveform + sem_waveform,
                        alpha=0.2)
    ax.plot(time_axis, avg_waveform)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Amplitude / microV")
    ax.set_title("Average Waveform of Detected Events")

    return ax

def get_phase_dist(result):
    """
    Takes a SimulationResult object and gets phase distribution
    of stimulation times.

    :param axs: SimulationResult
    """
    stim_phases = result.compute_stim_phase()

    return stim_phases

def plot_phase_dist(ax,
                    phases,
                    target_phase):
    
    mean_phase = scipy.stats.circmean(phases)
    std_phase = scipy.stats.circstd(phases)

    ax.hist(phases,
             bins=30,
             range=(0, 2 * pi),
             color='slateblue',
             edgecolor='black',
             linewidth=0.5)    
    r_max = 1.2 * ax.get_ylim()[1]

    ax.plot([mean_phase] * 2, [0, r_max],
             color='red',
             linestyle='--',
             linewidth=1,
             label='mean phase')

    ax.plot([target_phase] * 2, [0, r_max],
             color='black',
             linestyle='--',
             linewidth=1,
             label='target phase')
    
    ax.set_xticks([0, pi / 2, pi, 3 * pi / 2])
    ax.set_xticklabels(
        ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
    ax.set_yticks(ax.get_yticks()[-1:])
    ax.grid(True, alpha=0.2)
    ax.set_title(f'"Average Phase of Slow Wave at Stimulation"\nMean: {mean_phase:.2f}, Std: {std_phase:.2f}')

    return ax

def get_false_positives(stims,
                        ieds,
                        buffer_s=1):
    """
    Takes a SimulationResult object and
    gets proportion of IEDs that were falsely detected.
    """
    if len(ieds) == 0 or len(stims) == 0:
        return 0
    
    false_detect_ieds = [i for i in ieds if
                         np.min(np.abs(stims - i)) < buffer_s]
    
    prop_false_ieds = len(false_detect_ieds) / len(ieds)

    return prop_false_ieds

def plot_fp_prop_hist(ax,
                      fp_props):

    ax.hist(fp_props, bins=np.arange(0, 1.1, 0.05), 
             color='skyblue', edgecolor='black', alpha=0.8)

    # Add a vertical line for the mean
    ax.axvline(np.mean(fp_props), color='red', linestyle='--', 
                label=f'mean precision: {np.mean(fp_props):.3f}')

    ax.set_title("distribution of falsely stimulated IEDs across all channels (tol = 1.0s)")
    ax.set_xlabel("precision")
    ax.set_ylabel("count (number of channels)")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return ax

def plot_master(axs,
                data_dir="data/annotated",
                out_dir="results/run_all/run40"):
    """
    Takes an axes object and fills it with an average waveform plot
    according to the data in the passed directory.
    
    :param axs: Tuple of matplotlib.axes.Axes
    :param data_fir: str
    :param out_dir: str
    """
    p_c_struct = get_p_c_struct(data_dir)

    all_ps_wavs = []
    all_ps_phases = []
    all_ps_fp_props = []

    for p, cs in p_c_struct.items():

        print(f"starting patient {p}")
        # if int(p) > 7:
        #     continue
        for c in cs:

            result_filename = f"results_twave_all_p{int(p)}_c{c}.pkl"
            result_filepath = os.path.join(out_dir, result_filename)
            with open(result_filepath, "rb") as f:
                this_result = pickle.load(f)
            this_times = np.array(this_result.stims_sp) / this_result.Dataset.fs
            this_signal = this_result.Dataset.signal

            ied_filename = f"Patient{p}_Channel{c}_IEDs.npy"
            ied_path = os.path.join("data/annotated", ied_filename)
            this_ieds = np.load(ied_path)
            this_ieds = np.array([x for x in this_ieds if x <= this_result.Dataset.t.max()])

            # extend avg waveform list
            all_ps_wavs.extend(get_stim_waveforms(this_signal, this_times))

            # extend phase dist list
            all_ps_phases.extend(get_phase_dist(this_result))

            # append false positive prop to list
            all_ps_fp_props.append(get_false_positives(this_times, this_ieds))


    # plot average waveform
    plot_avg_waveform(axs[0], all_ps_wavs)

    # plot phase distribution
    plot_phase_dist(axs[1], all_ps_phases, target_phase=0)

    # plot detection precision
    plot_fp_prop_hist(axs[2], all_ps_fp_props)

# %% MAIN SCRIPT

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1) # Normal
ax2 = fig.add_subplot(1, 3, 2, projection='polar') # Polar for Phase
ax3 = fig.add_subplot(1, 3, 3) # Normal

plot_master(axs=[ax1, ax2, ax3])

plt.tight_layout()
plt.show()


# %%
