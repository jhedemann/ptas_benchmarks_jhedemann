# %% IMPORTS

import pandas as pd
import numpy as np
import mne
import mne_bids
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from utils.Simulations import SimulationDataset
except ModuleNotFoundError:
    from Simulations import SimulationDataset

# %% METHODS

def load_eeg_bids(subject, session, task, acquisition,
                  channel,
                  datatype="eeg",
                  root="/home/jhedemann/ptas_benchmarks_jhedemann/data/ear-eeg",
                  fs=250,
                  name=None,
                  max_duration_s=None,
                  data_units="V"):
    """
    Load EEG data as BIDS object and return arrays compatible with the ptas pipeline.
    """

    data_path_bids = mne_bids.BIDSPath(subject=subject,
                                       session=session,
                                       task=task,
                                       acquisition=acquisition,
                                       datatype=datatype,
                                       root=root)
    
    eeg_raw_bids = mne_bids.read_raw_bids(data_path_bids)
    eeg_raw = eeg_raw_bids.get_data()

    if max_duration_s is None:
        sig = eeg_raw[channel,:]
    else:
        sig = eeg_raw[channel, :int(max_duration_s * fs)]

    t = np.arange(sig.size) / fs

    # units handling
    if data_units.lower() in ("v", "volt", "volts"):
        sig_uv = sig * 1e6
    elif data_units.lower() in ("uv", "µv", "microvolt", "microvolts"):
        sig_uv = sig
    else:
        raise ValueError("data_units must be 'uV' or 'V'")

    if name is None:
        name = f"sub-{subject}-ses-{session}-c-{channel}"

    return t, sig_uv, name

def load_eeg_as_dataset(subject, session, task,
                        channel,
                        acquisition="earEEG",
                        datatype="eeg",
                        root="/home/jhedemann/ptas_benchmarks_jhedemann/data/ear-eeg",
                        fs=250,
                        max_duration_s=None):
    
    t, sig_uv, name = load_eeg_bids(subject=subject,
                              session=session,
                              task=task,
                              acquisition=acquisition,
                              channel=channel,
                              datatype=datatype,
                              root=root,
                              fs=fs,
                              max_duration_s=max_duration_s,
                              data_units="V")
    
    return SimulationDataset(t=t, signal=sig_uv, fs=fs, name=name)


# %% MAIN SCRIPT

if __name__ == "__main__":
    
    ds = load_eeg_as_dataset(subject="001",
                             session="001",
                             task="sleep",
                             channel=0)
    print(ds.name, ds.fs, ds.signal.shape, ds.signal.min(), ds.signal.max())


    data_path = "data/ear-eeg/sub-001/ses-001/eeg/sub-001_ses-001_task-sleep_acq-earEEG_eeg.set"
    bids_root = "/home/jhedemann/ptas_benchmarks_jhedemann/data/ear-eeg"

    data_path_bids = mne_bids.BIDSPath(subject="001",
                                    session="001",
                                    task="sleep",
                                    acquisition="earEEG",
                                    datatype="eeg",
                                    root=bids_root)

    eeg_ppt001_ses_001 = mne_bids.read_raw_bids(data_path_bids)

    time_axis = np.arange(0, 15000/250, 1/250)
    eeg = eeg_ppt001_ses_001.get_data()
    plt.plot(time_axis, eeg[0, :15000]*1000000, alpha=0.2, label="channel 0")
    plt.plot(time_axis, eeg[1, :15000]*1000000, alpha=0.2, label="channel 1")
    plt.plot(time_axis, eeg[2, :15000]*1000000, alpha=0.2, label="channel 2")
    plt.plot(time_axis, eeg[3, :15000]*1000000, alpha=0.2, label="channel 3")
    plt.plot(time_axis, eeg[4, :15000]*1000000, alpha=0.2, label="channel 4")
    plt.legend()

    plt.show()
