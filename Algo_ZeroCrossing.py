import numpy as np
from Simulations import PhaseTrackerResult, PhaseTrackerStatus
from typing import Tuple
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi

# ===============================
# Adaptive amplitude estimator
# ===============================
class RollingQuantileThreshold:
    def __init__(self, q=0.05, maxlen=3000, warmup=50,
                 floor_uv=-100, ceil_uv=-5000):
        self.q = q
        self.buf = deque(maxlen=maxlen)
        self.warmup = warmup
        self.floor_uv = floor_uv
        self.ceil_uv = ceil_uv

    def update(self, trough_uv: float):
        self.buf.append(float(trough_uv))

    def value(self, fallback_uv: float) -> float:
        if len(self.buf) < self.warmup:
            return fallback_uv
        thr = float(np.quantile(np.asarray(self.buf), self.q))
        thr = min(thr, self.floor_uv)
        thr = max(thr, self.ceil_uv)
        return thr

# ===============================
# Zero-crossing phase tracker
# ===============================
class PhaseTracker:
    name = "ZeroCrossing"

    def __init__(
        self,
        fs: int = 512,
        min_peak_uv: float = -1500,
        min_interval_ms: int = 300,
        max_interval_ms: int = 1000,
        backoff_sp: int = 2500,
        stim_delay_sp: int = 0,
        interstim_sp: int = 1000,
        history_len: int = 350, # changed from 250, 2000
        amp_q: float = 0.2,
        amp_warmup: int = 50,
        amp_maxlen: int = 3000,
        amp_floor_uv: float = -100,
        amp_ceil_uv: float = -5000,
    ):
        self.fs = fs

        # amplitude thresholding
        self.base_min_peak_uv = min_peak_uv
        self.amp_est = RollingQuantileThreshold(
            q=amp_q,
            maxlen=amp_maxlen,
            warmup=amp_warmup,
            floor_uv=amp_floor_uv,
            ceil_uv=amp_ceil_uv,
        )

        # filtering for IEDs
        self.lookback = 0.15 # mean delay between positive spike and negative trough of IED
        self.filter_thr = 120 # uV, changed from 900, 300, 520
        # self.high_sos = butter(4, [20, 80], btype="bandpass", fs=fs, output="sos")
        self.high_sos = butter(4, 90, btype="highpass", fs=fs, output="sos")
        self.high_zi = sosfilt_zi(self.high_sos)
        self._high_filtered_history = deque(maxlen=history_len)
        self.low_sos = butter(4, [0.5, 4.0], btype="bandpass", fs=fs, output="sos")
        self.low_zi = sosfilt_zi(self.low_sos)
        self._low_filtered_history = deque(maxlen=history_len)

        # timing
        self.min_interval_sp = int(min_interval_ms * fs / 1000)
        self.max_interval_sp = int(max_interval_ms * fs / 1000)
        self.backoff_sp = backoff_sp
        self.stim_delay_sp = stim_delay_sp
        self.interstim_sp = interstim_sp

        # state
        self._current_time_sp = 0
        self._last_stim_sp = -np.inf
        self._signal_history = deque(maxlen=history_len)
        self._last_value = None
        self._negzc_time = None
        self._neg_peak = None
        self._neg_peak_time = None
        self._awaiting_poszc = False

    def update(self, signal: float) -> Tuple[PhaseTrackerResult, dict]:
        self._signal_history.append(signal)
        self._current_time_sp += 1
        internals = {'phase': np.nan}

        # low frequency band filter
        low_signal, self.low_zi = sosfilt(self.low_sos, [signal], zi=self.low_zi)
        low_signal = low_signal[0]
        self._low_filtered_history.append(low_signal)

        # backoff / ISI logic
        if self._current_time_sp - self._last_stim_sp == self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.STIM2), internals
        if self._current_time_sp - self._last_stim_sp < self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF_ISI), internals
        if self._current_time_sp - self._last_stim_sp < self.backoff_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF), internals

        if self._last_value is not None:

            # negative-going ZC
            if self._last_value > 0 and low_signal <= 0:
                self._negzc_time = self._current_time_sp - 1
                self._neg_peak = low_signal
                self._neg_peak_time = self._current_time_sp - 1
                self._awaiting_poszc = True

            elif self._awaiting_poszc:

                # track trough
                if low_signal < self._neg_peak:
                    self._neg_peak = low_signal
                    self._neg_peak_time = self._current_time_sp - 1

                # positive-going ZC
                if self._last_value < 0 and low_signal >= 0:
                    poszc_time = self._current_time_sp - 1
                    interval = poszc_time - self._negzc_time

                    # learn amplitude distribution
                    if self.min_interval_sp <= interval <= self.max_interval_sp:
                        self.amp_est.update(self._neg_peak)

                    amp_thr = self.amp_est.value(self.base_min_peak_uv)
                    internals["amp_thr"] = amp_thr

                    # high frequency band filter
                    high_filtered_sample, self.high_zi = sosfilt(self.high_sos, [signal], zi=self.high_zi)
                    self._high_filtered_history.append(high_filtered_sample[0])
                    hf_window = list(self._high_filtered_history)

                    if np.max(hf_window) > self.filter_thr:
                        print(f"max filtered amp at this point is {np.max(hf_window)}")

                    # final SW decision
                    if (
                        self._neg_peak <= amp_thr
                        and self.min_interval_sp <= interval <= self.max_interval_sp
                        and np.max(hf_window) <= self.filter_thr
                    ):
                        self._last_stim_sp = self._current_time_sp
                        self._awaiting_poszc = False
                        internals.update(
                            neg_peak=self._neg_peak,
                            neg_peak_time=self._neg_peak_time,
                            interval=interval,
                        )
                        return PhaseTrackerResult(
                            PhaseTrackerStatus.STIM1, self.stim_delay_sp
                        ), internals

                    self._awaiting_poszc = False

        self._last_value = low_signal
        return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE), internals
