
import numpy as np
import scipy.signal
from collections import deque
from typing import Tuple, Dict, Optional

# IMPORTANT: use the *same* status/result classes as Simulations.py
from Simulations import PhaseTrackerResult, PhaseTrackerStatus


class _OnlineSOSFilter:
    """Streaming SOS filter with state."""
    def __init__(self, sos: np.ndarray):
        self.sos = np.asarray(sos, dtype=float)
        self.zi = scipy.signal.sosfilt_zi(self.sos) * 0.0

    def step(self, x: float) -> float:
        y, self.zi = scipy.signal.sosfilt(self.sos, [x], zi=self.zi)
        return float(y[0])


class _OnlineEWStats:
    """Exponentially-weighted running mean/var (for online thresholds)."""
    def __init__(self, alpha: float, init_mean: float = 0.0, init_var: float = 1.0):
        self.alpha = float(alpha)
        self.mean = float(init_mean)
        self.var = float(init_var)

    def update(self, x: float):
        a = self.alpha
        dx = x - self.mean
        self.mean += a * dx
        # EW variance update (stable enough for gating)
        self.var = (1 - a) * (self.var + a * dx * dx)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-12)))


class _OnlineIEDRejector:
    """
    Online IED/spike gate:
      - high-band (20–80 Hz) envelope threshold (EW mean/std)
      - slope gate on the slow-wave band
      - refractory window suppresses detections
    """
    def __init__(self, fs: float,
                 env_k: float = 6.0,
                 refractory_s: float = 0.25,
                 env_alpha: float = 0.005):
        self.fs = float(fs)
        self.refractory_sp = int(round(refractory_s * fs))
        self.env_k = float(env_k)

        sos = scipy.signal.butter(
            2, [20/(fs/2), 80/(fs/2)], btype="bandpass", output="sos"
        )
        self.hf = _OnlineSOSFilter(sos)
        self.env_stats = _OnlineEWStats(alpha=env_alpha, init_mean=0.0, init_var=1.0)
        self._refr = 0

    def step(self, x_raw: float, slow_bp: float, slow_prev: float) -> bool:
        """Returns True if we should reject (IED-like)."""
        # high-band envelope
        hf = self.hf.step(x_raw)
        env = abs(hf)
        self.env_stats.update(env)
        thr = self.env_stats.mean + self.env_k * self.env_stats.std

        # slope gate on slow band (captures very sharp transients)
        slope = abs(slow_bp - slow_prev)

        # trigger reject if envelope huge or slope huge (slope is in same units as signal/sample)
        if env > thr or slope > 8.0 * (thr + 1e-12):
            self._refr = self.refractory_sp

        if self._refr > 0:
            self._refr -= 1
            return True
        return False


class PhaseTracker:
    """
    Online Laurent-style negative slow-wave detector (causal).

    Key behaviors:
      - ONLY targets negative-deflecting slow waves (negative half-wave trough)
      - Uses duration constraints on the negative half-wave (between zero crossings)
      - Uses an adaptive *strict* negative threshold: thr = max(floor, mean - k*std) on troughs
      - Rejects IED-like segments online (HF envelope + slope gate) and suppresses detections for a refractory period
      - Returns Simulations.PhaseTrackerResult/Status so Simulations.run_simulations works.
    """
    name = "LaurentOnline"

    def __init__(
        self,
        fs: int = 512,
        bp_low_hz: float = 0.5,
        bp_high_hz: float = 4.0,
        # negative half-wave duration bounds
        min_interval_ms: int = 250,
        max_interval_ms: int = 1200,
        # backoff/ISI (same semantics as Algo_ZeroCrossing)
        backoff_sp: int = 2500,
        interstim_sp: int = 1000,
        stim_delay_sp: int = 0,
        # thresholding on troughs (in signal units, typically µV)
        neg_floor_uv: float = -1000.0,   # absolute minimum: trough must be <= this, changed from -1200, -800, -80
        neg_k: float = 3.0,            # stricter than mean: mean - k*std, changed from 2, 1.5
        thr_alpha: float = 0.01,       # EW speed
        # IED rejection
        ied_env_k: float = 6.0,
        ied_refractory_s: float = 0.25,
    ):
        self.fs = int(fs)
        self.min_interval_sp = int(round(min_interval_ms * fs / 1000))
        self.max_interval_sp = int(round(max_interval_ms * fs / 1000))

        self.backoff_sp = int(backoff_sp)
        self.interstim_sp = int(interstim_sp)
        self.stim_delay_sp = int(stim_delay_sp)

        # slow-wave bandpass
        sos_sw = scipy.signal.butter(
            2, [bp_low_hz/(fs/2), bp_high_hz/(fs/2)], btype="bandpass", output="sos"
        )
        self.sw = _OnlineSOSFilter(sos_sw)

        # trough threshold stats (track trough values; they are negative)
        self.neg_floor_uv = float(neg_floor_uv)
        self.neg_k = float(neg_k)
        self.trough_stats = _OnlineEWStats(alpha=thr_alpha, init_mean=neg_floor_uv, init_var=100.0)

        # IED rejector works on raw input + slow-band slope
        self.ied = _OnlineIEDRejector(fs=float(fs), env_k=ied_env_k, refractory_s=ied_refractory_s)

        # streaming state
        self._t_sp = 0
        self._last_stim_sp = -10**9
        self._slow_prev = 0.0

        # zero-crossing / negative half-wave tracking
        self._in_neg_half = False
        self._zc_start_sp: Optional[int] = None
        self._trough_val: float = 0.0
        self._trough_sp: Optional[int] = None
        self._prev_sw = 0.0

    def _trough_threshold(self) -> float:
        # strict negative threshold: mean - k*std, but never above floor (i.e., never too permissive)
        thr = self.trough_stats.mean - self.neg_k * self.trough_stats.std
        thr = min(thr, self.neg_floor_uv)  # e.g., floor -80 => thr <= -80
        return float(thr)

    def update(self, signal: float) -> Tuple[PhaseTrackerResult, Dict]:
        """
        Online step. Returns (PhaseTrackerResult, internals)
        """
        self._t_sp += 1
        internals: Dict = {"phase": np.nan}

        # ISI/backoff logic identical to ZeroCrossing expectations
        if self._t_sp - self._last_stim_sp == self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.STIM2), internals
        if self._t_sp - self._last_stim_sp < self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF_ISI), internals
        if self._t_sp - self._last_stim_sp < self.backoff_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF), internals

        # slow-wave band
        sw = self.sw.step(signal)

        # IED rejection gate (causal)
        reject = self.ied.step(x_raw=signal, slow_bp=sw, slow_prev=self._slow_prev)
        self._slow_prev = sw

        # detect zero crossings on slow-wave band
        prev = self._prev_sw
        self._prev_sw = sw

        # Start negative half-wave: crossing from + to -
        if (not self._in_neg_half) and prev > 0 and sw <= 0:
            self._in_neg_half = True
            self._zc_start_sp = self._t_sp
            self._trough_val = sw
            self._trough_sp = self._t_sp

        # Track trough while in negative half-wave
        if self._in_neg_half:
            if sw < self._trough_val:
                self._trough_val = sw
                self._trough_sp = self._t_sp

        # End negative half-wave: crossing from - to +
        fired = False
        if self._in_neg_half and prev < 0 and sw >= 0:
            zc_end_sp = self._t_sp
            dur_sp = zc_end_sp - (self._zc_start_sp or zc_end_sp)

            # update trough stats with every completed negative half-wave (even rejected)
            self.trough_stats.update(self._trough_val)

            thr = self._trough_threshold()

            ok_dur = self.min_interval_sp <= dur_sp <= self.max_interval_sp
            ok_pol = self._trough_val < 0.0  # negative-deflecting only
            ok_amp = self._trough_val <= thr  # more negative than threshold
            ok_ied = (not reject)

            internals.update({
                "sw": sw,
                "dur_sp": dur_sp,
                "trough": self._trough_val,
                "thr": thr,
                "reject_ied": reject,
            })

            if ok_dur and ok_pol and ok_amp and ok_ied:
                fired = True

            # reset half-wave state
            self._in_neg_half = False
            self._zc_start_sp = None
            self._trough_sp = None
            self._trough_val = 0.0

        if fired:
            self._last_stim_sp = self._t_sp
            return PhaseTrackerResult(PhaseTrackerStatus.STIM1, delay_sp=self.stim_delay_sp), internals

        return PhaseTrackerResult(PhaseTrackerStatus.NONE), internals
