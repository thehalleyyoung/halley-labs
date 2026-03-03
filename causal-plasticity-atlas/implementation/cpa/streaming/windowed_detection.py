"""Windowed tipping-point detection.

Sliding-window change-point detection for streaming plasticity
signals, supporting CUSUM, Page-Hinkley, and cost-based (PELT-like)
detection methods.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DetectionWindow:
    """Metadata for a single detection window."""

    start_idx: int
    end_idx: int
    statistic: float
    is_changepoint: bool


# ---------------------------------------------------------------------------
# OnlineCUSUM
# ---------------------------------------------------------------------------

class OnlineCUSUM:
    """Online CUSUM (cumulative sum) detector for a single stream.

    Maintains upper and lower CUSUM statistics and signals a change
    when either exceeds *threshold*.

    Parameters
    ----------
    threshold : float
        Decision boundary for declaring a change.
    drift : float
        Allowance parameter (δ); the expected shift size divided by 2.
    """

    def __init__(
        self, threshold: float = 5.0, drift: float = 0.5
    ) -> None:
        self._threshold = threshold
        self._drift = drift
        self._s_upper = 0.0
        self._s_lower = 0.0
        self._mean = 0.0
        self._n = 0
        self._m2 = 0.0
        self._change_detected = False
        self._change_value = 0.0

    # -- public API --------------------------------------------------------

    def update(self, value: float) -> bool:
        """Process a single *value*.

        Returns ``True`` if a change was detected on this step.
        """
        self._n += 1
        # Update running mean (Welford)
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)

        residual = value - self._mean
        self._s_upper = self._upper_cusum(residual)
        self._s_lower = self._lower_cusum(residual)

        if self._s_upper > self._threshold or self._s_lower > self._threshold:
            self._change_detected = True
            self._change_value = max(self._s_upper, self._s_lower)
            return True
        return False

    def has_change(self) -> bool:
        """Return ``True`` if a change has been detected since last reset."""
        return self._change_detected

    def reset(self) -> None:
        """Reset after detection to begin monitoring for the next change."""
        self._s_upper = 0.0
        self._s_lower = 0.0
        self._change_detected = False
        self._change_value = 0.0

    def full_reset(self) -> None:
        """Reset all internal state including running mean."""
        self.reset()
        self._mean = 0.0
        self._n = 0
        self._m2 = 0.0

    # -- CUSUM statistics --------------------------------------------------

    def _upper_cusum(self, residual: float) -> float:
        """Upper CUSUM statistic: detects upward shifts."""
        self._s_upper = max(0.0, self._s_upper + residual - self._drift)
        return self._s_upper

    def _lower_cusum(self, residual: float) -> float:
        """Lower CUSUM statistic: detects downward shifts."""
        self._s_lower = max(0.0, self._s_lower - residual - self._drift)
        return self._s_lower

    @property
    def statistics(self) -> Tuple[float, float]:
        """Return ``(upper_cusum, lower_cusum)``."""
        return self._s_upper, self._s_lower

    def __repr__(self) -> str:
        return (
            f"OnlineCUSUM(threshold={self._threshold}, "
            f"n={self._n}, change={self._change_detected})"
        )


# ---------------------------------------------------------------------------
# Page-Hinkley test (standalone function used by WindowedDetector)
# ---------------------------------------------------------------------------

def _page_hinkley(
    values: NDArray,
    threshold: float = 10.0,
    alpha: float = 0.005,
) -> List[int]:
    """Page-Hinkley test for change detection.

    Parameters
    ----------
    values : 1-D array
        Sequential observations.
    threshold : float
        Decision threshold λ.
    alpha : float
        Tolerance for the magnitude of change (δ).

    Returns
    -------
    changepoints : list of int
        Indices where changes were detected.
    """
    n = len(values)
    if n < 3:
        return []

    changepoints: List[int] = []
    cumsum = 0.0
    min_cumsum = 0.0
    running_mean = 0.0

    for t in range(n):
        running_mean = (running_mean * t + values[t]) / (t + 1)
        cumsum += values[t] - running_mean - alpha
        min_cumsum = min(min_cumsum, cumsum)

        if cumsum - min_cumsum > threshold:
            changepoints.append(t)
            cumsum = 0.0
            min_cumsum = 0.0

    return changepoints


# ---------------------------------------------------------------------------
# CUSUM batch detector (standalone)
# ---------------------------------------------------------------------------

def _cusum_detector(
    values: NDArray,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> List[int]:
    """Run CUSUM change-point detection over a batch of *values*.

    Returns indices of detected change points.
    """
    detector = OnlineCUSUM(threshold=threshold, drift=drift)
    changepoints: List[int] = []
    for i, v in enumerate(values):
        if detector.update(float(v)):
            changepoints.append(i)
            detector.reset()
    return changepoints


# ---------------------------------------------------------------------------
# Cost-based (PELT-like) detector
# ---------------------------------------------------------------------------

def _pelt_detector(
    values: NDArray,
    penalty: float = 1.0,
    min_segment: int = 3,
) -> List[int]:
    """Binary-segmentation change-point detection.

    Recursively splits the signal at the point of maximum cost
    reduction, stopping when the penalty exceeds the improvement.

    Parameters
    ----------
    values : 1-D array
        Sequential observations.
    penalty : float
        Per-changepoint penalty (BIC-style).
    min_segment : int
        Minimum distance between consecutive changepoints.

    Returns
    -------
    list of int
        Changepoint indices (sorted).
    """
    n = len(values)
    if n < 2 * min_segment:
        return []

    def _segment_cost(seg: NDArray) -> float:
        m = len(seg)
        if m < 2:
            return 0.0
        var = float(np.var(seg, ddof=1))
        if var < 1e-15:
            return 0.0
        return m * np.log(var + 1e-15)

    def _binary_seg(
        arr: NDArray, offset: int, results: List[int]
    ) -> None:
        m = len(arr)
        if m < 2 * min_segment:
            return
        full_cost = _segment_cost(arr)
        best_gain = -np.inf
        best_t = -1
        for t in range(min_segment, m - min_segment + 1):
            left_cost = _segment_cost(arr[:t])
            right_cost = _segment_cost(arr[t:])
            gain = full_cost - left_cost - right_cost - penalty
            if gain > best_gain:
                best_gain = gain
                best_t = t
        if best_gain > 0 and best_t > 0:
            results.append(offset + best_t)
            _binary_seg(arr[:best_t], offset, results)
            _binary_seg(arr[best_t:], offset + best_t, results)

    changepoints: List[int] = []
    _binary_seg(values, 0, changepoints)
    changepoints.sort()
    return changepoints


# ---------------------------------------------------------------------------
# WindowedDetector
# ---------------------------------------------------------------------------

class WindowedDetector:
    """Windowed tipping-point detector for streaming data.

    Ingests observations one at a time and runs changepoint detection
    over sliding windows.

    Parameters
    ----------
    window_size : int
        Number of observations per window.
    step_size : int
        Stride between successive windows.
    detection_method : str
        Algorithm: ``"cusum"``, ``"page_hinkley"``, or ``"pelt"``.
    threshold : float
        Significance / decision threshold.
    """

    _METHODS = {
        "cusum": "_run_cusum",
        "page_hinkley": "_run_page_hinkley",
        "pelt": "_run_pelt",
    }

    def __init__(
        self,
        window_size: int = 50,
        step_size: int = 10,
        detection_method: str = "pelt",
        threshold: float = 0.05,
    ) -> None:
        self._window_size = window_size
        self._step_size = step_size
        self._detection_method = detection_method
        self._threshold = threshold
        self._buffer: List[NDArray] = []
        self._detected: List[DetectionWindow] = []
        self._global_idx = 0
        self._last_processed = 0
        self._online_cusum: Optional[OnlineCUSUM] = None
        if detection_method == "cusum":
            self._online_cusum = OnlineCUSUM(
                threshold=threshold * 100, drift=0.5
            )

    # -- public API --------------------------------------------------------

    def feed(self, data_point: NDArray) -> None:
        """Ingest a single observation into the buffer."""
        self._buffer.append(np.asarray(data_point, dtype=np.float64))
        self._global_idx += 1

        # Trigger window-based detection when enough data accumulated
        while (
            self._last_processed + self._window_size <= len(self._buffer)
            and len(self._buffer) - self._last_processed >= self._step_size
        ):
            self._process_window()

    def detect(self) -> List[DetectionWindow]:
        """Run detection over any remaining buffered windows."""
        while (
            self._last_processed + self._window_size <= len(self._buffer)
        ):
            self._process_window()
        return list(self._detected)

    def current_window(self) -> NDArray:
        """Return the data in the current (possibly incomplete) window."""
        if not self._buffer:
            return np.empty(0, dtype=np.float64)
        start = max(0, len(self._buffer) - self._window_size)
        window_data = self._buffer[start:]
        if window_data[0].ndim == 0:
            return np.array(window_data, dtype=np.float64)
        return np.vstack(window_data)

    def reset(self) -> None:
        """Clear all internal state and detection history."""
        self._buffer.clear()
        self._detected.clear()
        self._global_idx = 0
        self._last_processed = 0
        if self._online_cusum is not None:
            self._online_cusum.full_reset()

    def detected_changepoints(self) -> List[int]:
        """Return global indices of all detected change points."""
        return [
            dw.start_idx + (dw.end_idx - dw.start_idx) // 2
            for dw in self._detected
            if dw.is_changepoint
        ]

    def get_detections(self) -> List[DetectionWindow]:
        """Return all :class:`DetectionWindow` objects."""
        return list(self._detected)

    def process(
        self, timestamp: int, descriptor: NDArray
    ) -> Optional[DetectionWindow]:
        """Feed a descriptor with explicit timestamp.

        Returns a :class:`DetectionWindow` if a change is detected
        in the current window, else ``None``.
        """
        self.feed(descriptor)
        if self._detected and self._detected[-1].is_changepoint:
            return self._detected[-1]
        return None

    # -- internal: window processing ---------------------------------------

    def _process_window(self) -> None:
        """Extract the next window and run detection."""
        start = self._last_processed
        end = start + self._window_size
        if end > len(self._buffer):
            return

        window = self._buffer[start:end]
        self._last_processed += self._step_size

        method_name = self._METHODS.get(
            self._detection_method, "_run_pelt"
        )
        method_fn = getattr(self, method_name)

        detections = method_fn(window, start)
        self._merge_detections(detections)

    # -- detection methods -------------------------------------------------

    def _run_cusum(
        self, window: List[NDArray], offset: int
    ) -> List[DetectionWindow]:
        """CUSUM detection over *window*."""
        values = self._flatten_window(window)
        cps = _cusum_detector(
            values, threshold=self._threshold * 100, drift=0.5
        )
        return self._cps_to_detections(cps, offset, len(values), "cusum")

    def _run_page_hinkley(
        self, window: List[NDArray], offset: int
    ) -> List[DetectionWindow]:
        """Page-Hinkley detection over *window*."""
        values = self._flatten_window(window)
        cps = _page_hinkley(values, threshold=self._threshold * 200)
        return self._cps_to_detections(cps, offset, len(values), "ph")

    def _run_pelt(
        self, window: List[NDArray], offset: int
    ) -> List[DetectionWindow]:
        """PELT-like cost-based detection over *window*."""
        values = self._flatten_window(window)
        penalty = self._threshold * np.log(len(values))
        cps = _pelt_detector(values, penalty=penalty, min_segment=3)
        return self._cps_to_detections(cps, offset, len(values), "pelt")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _flatten_window(window: List[NDArray]) -> NDArray:
        """Reduce a multivariate window to a univariate signal.

        Uses the L2 norm of each observation when multi-dimensional.
        """
        arr = np.array(window, dtype=np.float64)
        if arr.ndim > 1 and arr.shape[1] > 1:
            return np.linalg.norm(arr, axis=1)
        return arr.ravel()

    @staticmethod
    def _cps_to_detections(
        cps: List[int],
        offset: int,
        window_len: int,
        method: str,
    ) -> List[DetectionWindow]:
        """Convert changepoint indices to :class:`DetectionWindow`."""
        if not cps:
            return [
                DetectionWindow(
                    start_idx=offset,
                    end_idx=offset + window_len,
                    statistic=0.0,
                    is_changepoint=False,
                )
            ]
        results: List[DetectionWindow] = []
        for cp in cps:
            results.append(
                DetectionWindow(
                    start_idx=offset + cp,
                    end_idx=offset + window_len,
                    statistic=float(cp),
                    is_changepoint=True,
                )
            )
        return results

    def _merge_detections(
        self, new_detections: List[DetectionWindow]
    ) -> None:
        """Merge *new_detections* into ``self._detected``.

        De-duplicates nearby changepoints (within step_size).
        """
        for det in new_detections:
            if not det.is_changepoint:
                continue
            # Check if a nearby detection already exists
            duplicate = False
            for existing in self._detected:
                if (
                    existing.is_changepoint
                    and abs(existing.start_idx - det.start_idx)
                    < self._step_size
                ):
                    duplicate = True
                    break
            if not duplicate:
                self._detected.append(det)

    # -- update window -----------------------------------------------------

    def _update_window(
        self, timestamp: int, descriptor: NDArray
    ) -> None:
        """Add *descriptor* to the sliding window."""
        self.feed(descriptor)

    def _detect_in_window(
        self, window: List[NDArray]
    ) -> List[DetectionWindow]:
        """Run changepoint detection on a window."""
        method_name = self._METHODS.get(
            self._detection_method, "_run_pelt"
        )
        method_fn = getattr(self, method_name)
        return method_fn(window, 0)

    def __repr__(self) -> str:
        return (
            f"WindowedDetector(window={self._window_size}, "
            f"step={self._step_size}, method={self._detection_method!r}, "
            f"detections={len(self._detected)})"
        )
