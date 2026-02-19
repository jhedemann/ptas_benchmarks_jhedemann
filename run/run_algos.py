# %% IMPORTS

import numpy as np
from scipy.signal import butter, sosfiltfilt
import pickle

from utils.load_intracranial_data import load_data_as_dataset

import algos.Simulations
from algos.Algo_PLL import PhaseTracker as PLL
from algos.Algo_TWave import PhaseTracker as TWave
from algos.Algo_AmpTh import PhaseTracker as AmpTh
from algos.Algo_SineFit import PhaseTracker as SineFit
from algos.Algo_ZeroCrossing import PhaseTracker as ZeroCross

from algos.Inhibitors import *

# %%

ds = load_data_as_dataset(npy_path="/home/jhedemann/slow-wave/annotated/Patient03_Channel1_EEG.npy",
                          fs=1024)
x = ds.signal[:300*1024].squeeze().astype(float)

x = ds.signal.squeeze().astype(float)
sos = butter(4, [0.3, 4.0], btype="bandpass", fs=ds.fs, output="sos")
x_f = sosfiltfilt(sos, x)

# # downsample 1024 to 256 Hz
# x_ds = resample_poly(x_f, up=1, down=4)
# fs_ds = ds.fs // 4

# downsample 1024 to 128 Hz
# x_ds = resample_poly(x_f, up=1, down=8)
# fs_ds = ds.fs // 8

algo_names = ['PLL', 'TWave', 'AmpTh', 'SineFit', 'ZeroCrossing']
algos = {}

algos['PLL'] = PLL(
    fs=ds.fs,
    inhibitors=[MinAmp(window_length_sp=int(ds.fs * 2),
                       min_amp_threshold_uv=45)])

algos['TWave'] = TWave(ds.fs)

algos['AmpTh'] = AmpTh(stim_delay_sp=int(ds.fs * 0.5),
                       adaptive_window_sp=int(ds.fs * 5),
                       backoff_sp=int(ds.fs * 5))

algos['SineFit'] = SineFit(fs=ds.fs)
algos['ZeroCrossing'] = ZeroCross()

# # %% 

# # results = {}
# # for name, algo in algos.items():
# #     results[name] = Simulations.run_simulations(ds, algo)

# # results_twave = Simulations.run_simulations(ds, algos['TWave'])

# ds_short = load_data_as_dataset(npy_path="/home/jhedemann/slow-wave/1024hz/Patient04_electrode01.npy",
#                                 fs=1024,
#                                 max_duration_s=300)
# rslt = Simulations.run_simulations(ds_short, TWave(ds_short.fs))

# with open("results_twave.pkl", "wb") as f:
#     pickle.dump(rslt, f)

#%% 

print(ds.fs)

print("ds.fs:", ds.fs)
print("ds.signal.shape:", np.asarray(ds.signal).shape)
print("x.shape:", x.shape)
print("len(x):", len(x))
print("duration_s (from x):", len(x)/ds.fs)

# %%

ds2 = algos.Simulations.SimulationDataset(
    t=np.arange(len(x)) / ds.fs,
    signal=x,
    fs=ds.fs,
    name=ds.name + f"_bp03-4_ds{ds.fs}",
)

rslt = algos.Simulations.run_simulations(ds2, TWave(ds2.fs))

with open("results_twave_downsampled_128.pkl", "wb") as f:
    pickle.dump(rslt, f)

# %%

print(rslt.stims_sp)

# %%
