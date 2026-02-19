# %% IMPORTS

from pathlib import Path
import numpy as np
from Simulations import SimulationDataset

# %% FUNCTIONS

def load_intracranial_npy(path, fs, name=None, max_duration_s=None, data_units="uV"):
    """
    Load intracranial recording from .npy and return arrays compatible with the ptas pipeline.

    Supports shapes:
      - (n_samples,) 
      - (n_samples, 1)
      - (1, n_samples)
      - (n_channels, n_samples)  -> takes channel 0
      - (n_samples, n_channels)  -> takes channel 0
    """
    path = Path(path)
    arr = np.load(path)

    # normalize to 1D
    if arr.ndim == 1:
        sig = arr
    elif arr.ndim == 2:
        # for (n_samples, 1)
        if arr.shape[1] == 1:
            sig = arr[:, 0]
        # for (1, n_samples)
    elif arr.shape[0] == 1:
            sig = arr[0, :]
    else:
        raise ValueError(f"Unsupported array shape {arr.shape} in {path}")

    # optional if we do not want to load the entire dataset
    if max_duration_s is not None:
        sig = sig[: int(max_duration_s * fs)]

    t = np.arange(sig.shape[0]) / fs

    # units handling
    if data_units.lower() in ("v", "volt", "volts"):
        sig_uv = sig * 1e6
    elif data_units.lower() in ("uv", "µv", "microvolt", "microvolts"):
        sig_uv = sig
    else:
        raise ValueError("data_units must be 'uV' or 'V'")

    if name is None:
        name = path.stem

    return t, sig_uv

def load_data_as_dataset(npy_path, fs=1024, max_duration_s=None):
    t, sig_uv = load_intracranial_npy(
        npy_path,
        fs=fs,
        max_duration_s=max_duration_s,
        data_units="uV",   # your data appears to already be microvolts
    )
    return SimulationDataset(t=t, signal=sig_uv, fs=fs, name=Path(npy_path).stem)


def load_sw_annotation(npy_path):
    path = Path(npy_path)
    arr = np.load(path)
    print(len(arr))
    print(arr.shape)
    print(arr)
    return arr 

if __name__ == "__main__":
    # %%

    ds = load_data_as_dataset("/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_EEG.npy", fs=1024)
    print(ds.name, ds.fs, ds.signal.shape, ds.signal.min(), ds.signal.max())

    # %%

    x = ds.signal.squeeze()
    print("median(|x|):", np.median(np.abs(x)))
    print("p95(|x|):", np.percentile(np.abs(x), 95))
    print("p99(|x|):", np.percentile(np.abs(x), 99))
    print("min/max:", x.min(), x.max())

    # %%
    path_sw_annot = "/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_negSWs.npy"

    arr_sw_annot = load_sw_annotation(path_sw_annot)
    print(arr_sw_annot.shape)

    # %%