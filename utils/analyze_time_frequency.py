# %% IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from collections import defaultdict

from utils.load_intracranial_data import load_data_as_dataset

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

def filter_for_competing_events(events_1, events_2, buffer_s):

    if len(events_1) == 0 or len(events_2) == 0:
            return np.array(events_1), np.array(events_2)

    helper_1 = events_1
    helper_2 = events_2

    events_1 = [i for i in events_1 if
                    np.min(np.abs(helper_2 - i)) > buffer_s]
    events_2 = [i for i in events_2 if
                    np.min(np.abs(helper_1 - i)) > buffer_s]

    return np.array(events_1), np.array(events_2)

def find_maxima(signal, times, fs, window_size_s, low_pass=None):
    max_idx = []
    
    # low-pass filter signal such that maximum is not high-freq jitter
    if low_pass is not None:
        search_signal = mne.filter.filter_data(
            signal.astype(float), fs, l_freq=None, h_freq=low_pass, verbose=False
        )
    else:
        search_signal = signal

    idx = (times * fs).astype(int).ravel()
    window_size_idx = int(window_size_s * fs)
    sig_len = len(signal)

    for i in idx:

        # lookout for signal boundaries
        start = max(0, i - window_size_idx)
        end = min(sig_len, i + window_size_idx)
        
        window = search_signal[start:end]
        
        if len(window) == 0:
            continue
            
        max_i = np.argmax(window) + start
        max_idx.append(max_i)
    
    return np.array(max_idx, dtype=int)

def find_minima(signal, times, fs=512, window_size_s=4, low_pass=None):
    min_idx = []
    
    # low-pass filter signal such that maximum is not high-freq jitter
    if low_pass is not None:
        search_signal = mne.filter.filter_data(
            signal.astype(float), fs, l_freq=None, h_freq=low_pass, verbose=False
        )
    else:
        search_signal = signal

    idx = (times * fs).astype(int).ravel()
    window_size_idx = int(window_size_s * fs)
    sig_len = len(signal)

    for i in idx:

        # lookout for signal boundaries
        start = max(0, i - window_size_idx)
        end = min(sig_len, i + window_size_idx)
        
        window = search_signal[start:end]
        
        if len(window) == 0:
            continue
            
        min_i = np.argmin(window) + start
        min_idx.append(min_i)
    
    return np.array(min_idx, dtype=int)

def get_signal_subsets_from_events(signal, idx, fs=512, window_size_s=4):
    windows = []
    window_size_idx = int(window_size_s * fs)
    expected_length = 2 * window_size_idx
    sig_len = len(signal)

    for i in idx:
        i = int(i)
        # Define the theoretical boundaries
        start = i - window_size_idx
        end = i + window_size_idx
        
        # Define the actual boundaries (clamped to signal limits)
        actual_start = max(0, start)
        actual_end = min(sig_len, end)
        
        # Slice the available signal
        chunk = signal[actual_start:actual_end]
        
        if len(chunk) == expected_length:
            windows.append(chunk)
        else:
            # CREATE THE PADDING
            # 1. Start with an array of zeros of the exact target shape
            padded_window = np.zeros(expected_length)
            
            # 2. Calculate where to place the 'chunk' within the padded array
            # If start was -10, the chunk should start at index 10 of the padded array
            pad_start = actual_start - start 
            pad_end = pad_start + len(chunk)
            
            padded_window[pad_start:pad_end] = chunk
            windows.append(padded_window)
    
    return np.array(windows)

def find_empty_windows(sw_times, ied_times, fs, windows_size_s, empty_space_s, n_windows):

    # get all events into a sorted array
    total_times = np.concatenate([sw_times.ravel(), ied_times.ravel()])
    total_times = (total_times*fs).astype(int)
    total_times = np.sort(total_times)

    # keep appending intervals until next event is too close

    helper = np.insert(total_times, 0, 0)
    empty_windows = []
    len_interval = windows_size_s * fs
    safety_buffer = (empty_space_s - windows_size_s) / 2

    for i in range(len(helper)-1):

        safe_i = helper[i] + safety_buffer

        while (safe_i + len_interval + safety_buffer) < helper[i+1]:

            this_interval = (safe_i, safe_i+len_interval)
            empty_windows.append(this_interval)
            safe_i += len_interval

    return np.array(empty_windows).astype(int)

def get_signal_subsets_from_intervals(signal, intervals):

    signal_subsets = []

    for interval in intervals:

        start = interval[0]
        end = interval[1]

        this_subset = signal[start:end]
        signal_subsets.append(this_subset)
        
    return np.array(signal_subsets)

def do_event_tfr_on_all(in_dir: str,
                        freq_range: np.typing.NDArray[np.float64],
                        out_dir: str,
                        fs = 512,
                        comp_event_buffer_s = 4,
                        window_size_s = 4,
                        low_pass_filter_freq = 4,
                        empty_window_buffer_s = 12):
    
    # prep data for tfr

    p_c_struct = get_p_c_struct(in_dir)
    all_ps_sw_signals = []
    all_ps_ied_signals = []

    for p, cs in p_c_struct.items():

        this_p_sw_powers = []
        this_p_ied_powers = []
        this_p_non_powers = []

        this_p_sw_signals = []
        this_p_ied_signals = []

        for c in cs:

            signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
            signal_filepath = os.path.join(in_dir, signal_filename)

            sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
            sws_filepath = os.path.join(in_dir, sws_filename)

            ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
            ieds_filepath = os.path.join(in_dir, ieds_filename)

            dataset = load_data_as_dataset(signal_filepath, fs=fs)

            signal = dataset.signal

            arr_sws = np.load(sws_filepath)
            arr_ieds = np.load(ieds_filepath)

            filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=comp_event_buffer_s)

            sw_minima = find_minima(signal, filtered_arr_sw, fs, window_size_s, low_pass_filter_freq)
            sw_subsets = get_signal_subsets_from_events(signal, sw_minima, fs, window_size_s)
            sw_input = np.expand_dims(sw_subsets, axis=1)

            ied_maxima = find_minima(signal, filtered_arr_ied, fs, window_size_s, low_pass_filter_freq)
            ied_subsets = get_signal_subsets_from_events(signal, ied_maxima, fs, window_size_s)
            ied_input = np.expand_dims(ied_subsets, axis=1)

            this_p_sw_signals.append(sw_subsets)
            this_p_ied_signals.append(ied_subsets)

            total_intervals = find_empty_windows(sw_times=arr_sws,
                                        ied_times=arr_ieds,
                                        fs=fs,
                                        windows_size_s=window_size_s*2,
                                        empty_space_s=empty_window_buffer_s,
                                        n_windows=max((len(arr_sws), len(arr_ieds))))

            signal_without_events = get_signal_subsets_from_intervals(signal=signal,
                                                                      intervals=total_intervals)
            non_event_input = np.expand_dims(signal_without_events, axis=1)

            # run TFRs for this channel only

            if sw_input.size > 0:
                tfr_sw = mne.time_frequency.tfr_array_morlet(
                    sw_input, sfreq=fs, freqs=freq_range, n_cycles=5, output='power', verbose=False
                )
                avg_ch_sw_power = np.mean(tfr_sw, axis=0).squeeze()
                this_p_sw_powers.append(avg_ch_sw_power)
            else:
                print(f"Skipping SW TFR for P{p} C{c}: No events found.")

            if ied_input.size > 0:
                tfr_ied = mne.time_frequency.tfr_array_morlet(
                    ied_input, sfreq=fs, freqs=freq_range, n_cycles=5, output='power', verbose=False
                )
                avg_ch_ied_power = np.mean(tfr_ied, axis=0).squeeze()
                this_p_ied_powers.append(avg_ch_ied_power)
            else:
                print(f"Skipping IED TFR for P{p} C{c}: No events found.")

            if non_event_input.size > 0:
                tfr_non = mne.time_frequency.tfr_array_morlet(
                    non_event_input, sfreq=fs, freqs=freq_range, n_cycles=5, output='power', verbose=False
                )
                avg_ch_non_power = np.mean(tfr_non, axis=0).squeeze()
                this_p_non_powers.append(avg_ch_non_power)
            else:
                print(f"Skipping baseline TFR for P{p} C{c}: No events found.")

        all_ps_sw_signals.append(this_p_sw_signals)
        all_ps_ied_signals.append(this_p_ied_signals)

        this_p_avg_sw = np.mean(np.array(this_p_sw_powers), axis=0)
        this_p_avg_ied = np.mean(np.array(this_p_ied_powers), axis=0)
        this_p_avg_non = np.mean(np.array(this_p_non_powers), axis=0)
        
        baseline_vector = np.mean(this_p_avg_non, axis=1, keepdims=True)
        
        final_sw_tfr = np.log10(this_p_avg_sw) - np.log10(baseline_vector)
        final_ied_tfr = np.log10(this_p_avg_ied) - np.log10(baseline_vector)

        # Define the output path
        output_filename = f"Patient{p}_TFR_Results.npz"
        output_path = os.path.join(out_dir, output_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to disk
        np.savez_compressed(
            output_path, 
            sw_tfr=final_sw_tfr, 
            ied_tfr=final_ied_tfr,
            sw_signal=np.array(this_p_sw_signals, dtype=object),
            ied_signal=np.array(this_p_ied_signals, dtype=object),
            baseline_powers=baseline_vector.flatten(),
            freqs=freq_range,
            p_id=p
        )
        
        print(f"Successfully saved TFR for Patient {p}")

    return

def load_tfr_results(tfr_dir):
    """
    Loads all .npz files from a directory and returns a list 
    of dictionaries compatible with the plotting functions.
    """
    all_results = []
    
    # Get all .npz files in the directory
    files = [f for f in os.listdir(tfr_dir) if f.endswith('.npz')]
    files.sort() # Ensure consistent order
    
    for filename in files:
        path = os.path.join(tfr_dir, filename)
        
        with np.load(path, allow_pickle=True) as data:

            # Reconstruct the dictionary format
            p_dict = {
                "p_id": str(data['p_id']),
                "sw_tfr": data['sw_tfr'].copy() if 'sw_tfr' in data else None,
                "ied_tfr": data['ied_tfr'].copy() if 'ied_tfr' in data else None,
                "sw_signal": data['sw_signal'].copy() if 'sw_signal' in data else None,
                "ied_signal": data['ied_signal'].copy() if 'ied_signal' in data else None,
                "baseline_powers": data['baseline_powers'].copy() if 'baseline_powers' in data else None
            }

            all_results.append(p_dict)
            
    print(f"Successfully loaded {len(all_results)} participant results.")
    return all_results

def plot_single_tfr(p_results, freq_range, window_size_s=4):
    """
    p_results: dict containing 'sw_tfr' and 'ied_tfr'
    freq_range: the frequency array used in TFR
    """

    sw_data = p_results['sw_tfr']
    ied_data = p_results['ied_tfr']
    p_id = p_results.get('p_id', 'Unknown')

    sw_waveforms = np.array(p_results['sw_signal'], dtype=np.float64)
    ied_waveforms = np.array(p_results['ied_signal'], dtype=np.float64)

    # Global range for shared colorbar
    limit = max(np.abs(np.nanmin([sw_data, ied_data])), 
                np.abs(np.nanmax([sw_data, ied_data])))
    vmin, vmax = -limit, limit
    levels = np.linspace(vmin, vmax, 50)
    time_axis = np.linspace(-window_size_s, window_size_s, sw_data.shape[1])

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax1 = axs[0]
    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    axs[1].remove() # remove the placeholder
    axs[1] = ax2

    # Plot IEDs

    mean_ied_waveform = np.mean(ied_waveforms, axis=0)
    std_ied_waveform = np.std(ied_waveforms, axis=0)

    im0 = axs[0].contourf(time_axis, freq_range, ied_data, levels=levels, 
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    wvfrm0 = axs[0].twinx()
    wvfrm0.fill_between(time_axis, mean_ied_waveform - std_ied_waveform, mean_ied_waveform + std_ied_waveform,
                        color="black", alpha=0.05)
    wvfrm0.plot(time_axis, mean_ied_waveform, color="black", alpha=0.3)
    axs[0].set_title(f"Patient {p_id}: IEDs")

    # Plot SWs

    mean_sw_waveform = np.mean(sw_waveforms, axis=0)
    std_sw_waveform = np.std(sw_waveforms, axis=0)

    im1 = axs[1].contourf(time_axis, freq_range, sw_data, levels=levels, 
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    wvfrm1 = axs[1].twinx()
    wvfrm1.fill_between(time_axis, mean_sw_waveform - std_sw_waveform, mean_sw_waveform + std_sw_waveform,
                    color="black", alpha=0.05)
    wvfrm1.plot(time_axis, mean_sw_waveform, color="black", alpha=0.3)
    
    axs[1].set_title(f"Patient {p_id}: Slow Waves")

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(im1, ax=axs, label="Log Power Ratio (Event / Baseline)")
    plt.show()

    return

def get_grand_average_data(all_ps_results, freq_range):
    """
    Computes the robust mean across all participants and returns
    a single dictionary compatible with all plotting functions.
    """
    sw_list = [res['sw_tfr'] for res in all_ps_results if res['sw_tfr'] is not None]
    ied_list = [res['ied_tfr'] for res in all_ps_results if res['ied_tfr'] is not None]
    sw_signal = [res['sw_signal'] for res in all_ps_results if res['sw_signal'] is not None]
    ied_signal = [res['ied_signal'] for res in all_ps_results if res['ied_signal'] is not None]

    # Re-using your robust averaging logic
    def calculate_robust_average(data_list):
        if not data_list: return None
        target_freqs = len(freq_range)
        target_time = data_list[0].shape[1]
        sum_array = np.zeros((target_freqs, target_time))
        count = 0
        for arr in data_list:
            f_limit = min(target_freqs, arr.shape[0])
            t_limit = min(target_time, arr.shape[1])
            sum_array[:f_limit, :t_limit] += arr[:f_limit, :t_limit]
            count += 1
        return sum_array / count if count > 0 else None
    
    def unravel_nested_signals(signal_list):
        # Flatten the nested structure: Patients -> Channels -> Events -> Samples
        flat_list = []
        for p_data in signal_list:      # p_data is a list of channel data
            for c_data in p_data:       # c_data is a (n_events, n_samples) array
                if len(c_data) > 0:
                    flat_list.append(c_data)
        
        if not flat_list:
            return np.zeros(4096) # Default length if no data
            
        # Stack all events from all patients/channels into one big (TotalEvents, 4096) array
        all_events_combined = np.vstack(flat_list)
        
        # Mean across all events to get a 1D (4096,) waveform
        return all_events_combined

    grand_avg_results = {
        'p_id': 'GRAND AVERAGE',
        'sw_tfr': calculate_robust_average(sw_list),
        'ied_tfr': calculate_robust_average(ied_list),
        'sw_signal': unravel_nested_signals(sw_signal),
        'ied_signal': unravel_nested_signals(ied_signal),
    }
    return grand_avg_results

def plot_tfr_difference(p_results, freq_range, window_size_s=4):
    """
    Computes (SW_TFR - IED_TFR) and plots the contrast.
    Red = More power in Slow Waves
    Blue = More power in IEDs
    """
    sw_data = p_results['sw_tfr']
    ied_data = p_results['ied_tfr']
    p_id = p_results.get('p_id', 'Unknown')
    
    if sw_data is None or ied_data is None:
        print("Missing data for one of the event types.")
        return

    # Compute the difference
    diff_data = sw_data - ied_data

    # Setup time axis
    time_axis = np.linspace(-window_size_s, window_size_s, diff_data.shape[1])
    
    # Center the colorbar
    limit = np.nanmax(np.abs(diff_data))
    vmin, vmax = -limit, limit
    
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    cf = ax.contourf(time_axis, freq_range, diff_data, levels=50, 
                     vmin=vmin, vmax=vmax, cmap="RdBu_r")
    
    ax.set_title(f"TFR Contrast: Patient {p_id}\n(Slow Wave - IED)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    
    cbar = fig.colorbar(cf, ax=ax, label="Log Difference (Red: SW > IED | Blue: IED > SW)")
    
    plt.show()

    return

# %%

if __name__ == "__main__":

    freq_range=np.linspace(start=1, stop=200, num=50)

    # %% TFR ON ALL DATA

    # do_event_tfr_on_all(
    #     in_dir="data/annotated",
    #     freq_range=freq_range,
    #     out_dir="tfr/align_minima_both"
    #     )

    # %%

    results_list = load_tfr_results("tfr/align_minima_both")

    # # 2. Compute the Grand Average Data ONCE
    # grand_avg = get_grand_average_data(results_list, freq_range)

    # # 3. Plot the standard comparison (SW vs IED)
    # plot_single_tfr(grand_avg, freq_range)

    # # 4. Plot the difference (SW - IED)
    # plot_tfr_difference(grand_avg, freq_range)


    # Plot a 1/f curve
    baseline_powers = np.array([result["baseline_powers"] for result in results_list])
    print(baseline_powers.shape)
    baseline_powers_mean = np.mean(baseline_powers, axis=0)
    print(baseline_powers_mean)

    plt.plot(freq_range, np.log(baseline_powers_mean), color="black")
    plt.axvline(50, linestyle="--", color="red", label="line noise")
    plt.legend()
    plt.title("1/f Plot of Intracranial EEG")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Power (a.u.)")
    plt.show()