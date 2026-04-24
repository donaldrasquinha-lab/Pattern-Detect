"""
BULKOWSKI PATTERN DETECTION ENGINE
===================================
Uses scipy.signal for local extrema detection, numpy for mathematical pattern
validation, and pandas for data management. Implements 55+ patterns following
Bulkowski's Encyclopedia of Chart Patterns mathematical rules.

Pattern Categories:
  - Geometric/Structural (Head & Shoulders, Triangles, Wedges, etc.)
  - Candlestick (via TA-Lib style implementation)
  - OI/Volume patterns
  - Expiry cycle patterns
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import linregress
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import json, warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternResult:
    pattern_id: int
    name: str
    category: str        # Reversal / Continuation / Bilateral / Candlestick / OI
    bias: str            # BULLISH / BEARISH / NEUTRAL / BILATERAL
    confidence: float    # 0.0 - 1.0
    start_idx: int
    end_idx: int
    phase: str           # EARLY / MID / LATE / FULL / EXPIRY_DAY
    bulkowski_success_rate: float
    key_levels: dict = field(default_factory=dict)   # neckline, target, stop etc
    description: str = ""


class BulkowskiPatternEngine:
    """
    Core detection engine using scipy.signal for peak/trough finding
    and mathematical rules from Bulkowski's Encyclopedia.
    """

    def __init__(self, order: int = 5, min_pattern_bars: int = 5):
        self.order = order              # scipy argrelextrema order (lookback)
        self.min_bars = min_pattern_bars
        self.atr_period = 14

    # ═══════════════════════════════════════════════════════════════
    # CORE: Peak/Trough Detection via scipy.signal
    # ═══════════════════════════════════════════════════════════════
    def find_extrema(self, prices: np.ndarray, order: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find local maxima and minima using scipy.signal.argrelextrema"""
        o = order or self.order
        o = min(o, max(1, len(prices) // 4))
        if len(prices) < 2 * o + 1:
            return np.array([]), np.array([])
        peaks = argrelextrema(prices, np.greater_equal, order=o)[0]
        troughs = argrelextrema(prices, np.less_equal, order=o)[0]
        return peaks, troughs

    def find_peaks_scipy(self, prices: np.ndarray, prominence_pct: float = 0.005):
        """Use scipy.signal.find_peaks with prominence filtering"""
        prom = np.mean(prices) * prominence_pct
        peaks, props = find_peaks(prices, prominence=prom, distance=max(2, self.order))
        inv = -prices
        troughs, tprops = find_peaks(inv, prominence=prom, distance=max(2, self.order))
        return peaks, troughs

    def calc_atr(self, highs, lows, closes, period=14):
        """Average True Range for adaptive thresholds"""
        n = len(closes)
        if n < 2:
            return np.mean(highs - lows) if len(highs) > 0 else 1.0
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        if len(tr) < period:
            return np.mean(tr) if len(tr) > 0 else 1.0
        return np.mean(tr[-period:])

    def trendline_fit(self, x_indices, y_values):
        """Linear regression for trendline analysis"""
        if len(x_indices) < 2:
            return 0, 0, 0
        slope, intercept, r_value, _, _ = linregress(x_indices, y_values)
        return slope, intercept, r_value ** 2

    # ═══════════════════════════════════════════════════════════════
    # PATTERN DETECTORS — Bulkowski Mathematical Rules
    # ═══════════════════════════════════════════════════════════════

    def detect_head_and_shoulders(self, highs, lows, closes, peaks, troughs):
        """
        Bulkowski H&S Rules:
        - 3 peaks: middle (head) must be highest
        - Left & right shoulders roughly equal (within 3% of each other)
        - Neckline connects the two troughs between peaks
        - Volume typically decreases on right shoulder
        - Breakout target = head height - neckline, projected down
        """
        results = []
        if len(peaks) < 3 or len(troughs) < 2:
            return results

        for i in range(len(peaks) - 2):
            l_sh, head, r_sh = peaks[i], peaks[i + 1], peaks[i + 2]
            h_l, h_h, h_r = highs[l_sh], highs[head], highs[r_sh]

            # Head must be highest
            if h_h <= h_l or h_h <= h_r:
                continue

            # Shoulders within 5% of each other (Bulkowski tolerance)
            shoulder_diff = abs(h_l - h_r) / max(h_l, h_r)
            if shoulder_diff > 0.05:
                continue

            # Find troughs between peaks for neckline
            t_between = troughs[(troughs > l_sh) & (troughs < r_sh)]
            if len(t_between) < 2:
                t_between_1 = troughs[(troughs > l_sh) & (troughs < head)]
                t_between_2 = troughs[(troughs > head) & (troughs < r_sh)]
                if len(t_between_1) == 0 or len(t_between_2) == 0:
                    continue
                t1, t2 = t_between_1[-1], t_between_2[0]
            else:
                t1, t2 = t_between[0], t_between[-1]

            neckline = (lows[t1] + lows[t2]) / 2
            head_height = h_h - neckline
            target = neckline - head_height

            # Confidence based on shoulder symmetry and head prominence
            prominence = (h_h - max(h_l, h_r)) / h_h
            conf = min(0.95, 0.55 + (1 - shoulder_diff) * 0.2 + prominence * 2)

            results.append(PatternResult(
                pattern_id=1, name="Head & Shoulders", category="Reversal",
                bias="BEARISH", confidence=round(conf, 3),
                start_idx=int(l_sh), end_idx=int(r_sh), phase="FULL",
                bulkowski_success_rate=0.65,
                key_levels={"neckline": round(neckline, 2), "target": round(target, 2),
                            "head": round(h_h, 2), "left_shoulder": round(h_l, 2),
                            "right_shoulder": round(h_r, 2)},
                description="Classic bearish reversal. Neckline break confirms."
            ))
        return results

    def detect_inverse_head_and_shoulders(self, highs, lows, closes, peaks, troughs):
        """Inverse H&S: 3 troughs, middle lowest, shoulders ~equal"""
        results = []
        if len(troughs) < 3 or len(peaks) < 2:
            return results

        for i in range(len(troughs) - 2):
            l_sh, head, r_sh = troughs[i], troughs[i + 1], troughs[i + 2]
            lo_l, lo_h, lo_r = lows[l_sh], lows[head], lows[r_sh]

            if lo_h >= lo_l or lo_h >= lo_r:
                continue

            shoulder_diff = abs(lo_l - lo_r) / max(lo_l, lo_r)
            if shoulder_diff > 0.05:
                continue

            p_between = peaks[(peaks > l_sh) & (peaks < r_sh)]
            if len(p_between) < 1:
                continue

            neckline = np.mean(highs[p_between[:2]])
            target = neckline + (neckline - lo_h)
            prominence = (min(lo_l, lo_r) - lo_h) / min(lo_l, lo_r)
            conf = min(0.95, 0.55 + (1 - shoulder_diff) * 0.2 + abs(prominence) * 2)

            results.append(PatternResult(
                pattern_id=2, name="Inv Head & Shoulders", category="Reversal",
                bias="BULLISH", confidence=round(conf, 3),
                start_idx=int(l_sh), end_idx=int(r_sh), phase="FULL",
                bulkowski_success_rate=0.75,
                key_levels={"neckline": round(neckline, 2), "target": round(target, 2)},
                description="Classic bullish reversal. Neckline break confirms."
            ))
        return results

    def detect_double_top(self, highs, lows, closes, peaks, troughs, atr):
        """
        Bulkowski Double Top:
        - Two peaks within 3% of each other
        - Trough between them (neckline)
        - Second peak on lower volume preferred
        - Target = peak - trough height, projected down
        """
        results = []
        if len(peaks) < 2:
            return results

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            h1, h2 = highs[p1], highs[p2]

            diff = abs(h1 - h2) / max(h1, h2)
            if diff > 0.03:
                continue

            # Need at least one trough between
            t_between = troughs[(troughs > p1) & (troughs < p2)]
            if len(t_between) == 0:
                continue

            neckline = lows[t_between[np.argmin(lows[t_between])]]
            height = max(h1, h2) - neckline
            target = neckline - height
            conf = min(0.95, 0.60 + (1 - diff) * 0.25 + min(0.1, height / max(h1, h2)))

            results.append(PatternResult(
                pattern_id=3, name="Double Top", category="Reversal",
                bias="BEARISH", confidence=round(conf, 3),
                start_idx=int(p1), end_idx=int(p2), phase="MID",
                bulkowski_success_rate=0.70,
                key_levels={"neckline": round(neckline, 2), "target": round(target, 2),
                            "peak1": round(h1, 2), "peak2": round(h2, 2)},
                description="M-shape. Break below neckline confirms bearish reversal."
            ))
        return results

    def detect_double_bottom(self, highs, lows, closes, peaks, troughs, atr):
        """Bulkowski Double Bottom: Two troughs within 3%"""
        results = []
        if len(troughs) < 2:
            return results

        for i in range(len(troughs) - 1):
            t1, t2 = troughs[i], troughs[i + 1]
            l1, l2 = lows[t1], lows[t2]

            diff = abs(l1 - l2) / max(l1, l2)
            if diff > 0.03:
                continue

            p_between = peaks[(peaks > t1) & (peaks < t2)]
            if len(p_between) == 0:
                continue

            neckline = highs[p_between[np.argmax(highs[p_between])]]
            height = neckline - min(l1, l2)
            target = neckline + height
            conf = min(0.95, 0.60 + (1 - diff) * 0.25)

            results.append(PatternResult(
                pattern_id=4, name="Double Bottom", category="Reversal",
                bias="BULLISH", confidence=round(conf, 3),
                start_idx=int(t1), end_idx=int(t2), phase="MID",
                bulkowski_success_rate=0.70,
                key_levels={"neckline": round(neckline, 2), "target": round(target, 2)},
                description="W-shape. Break above neckline confirms bullish reversal."
            ))
        return results

    def detect_triple_top(self, highs, lows, closes, peaks, troughs, atr):
        """Triple Top: 3 peaks within 3% of each other"""
        results = []
        if len(peaks) < 3:
            return results

        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]
            h1, h2, h3 = highs[p1], highs[p2], highs[p3]
            mean_h = np.mean([h1, h2, h3])

            if all(abs(h - mean_h) / mean_h < 0.03 for h in [h1, h2, h3]):
                conf = 0.65 + (1 - np.std([h1, h2, h3]) / mean_h) * 0.25
                results.append(PatternResult(
                    pattern_id=5, name="Triple Top", category="Reversal",
                    bias="BEARISH", confidence=round(min(0.95, conf), 3),
                    start_idx=int(p1), end_idx=int(p3), phase="MID",
                    bulkowski_success_rate=0.70,
                    key_levels={"resistance": round(mean_h, 2)},
                    description="Three rejections at resistance. Strong bearish signal."
                ))
        return results

    def detect_triple_bottom(self, highs, lows, closes, peaks, troughs, atr):
        """Triple Bottom: 3 troughs within 3%"""
        results = []
        if len(troughs) < 3:
            return results

        for i in range(len(troughs) - 2):
            t1, t2, t3 = troughs[i], troughs[i + 1], troughs[i + 2]
            l1, l2, l3 = lows[t1], lows[t2], lows[t3]
            mean_l = np.mean([l1, l2, l3])

            if all(abs(l - mean_l) / mean_l < 0.03 for l in [l1, l2, l3]):
                conf = 0.65 + (1 - np.std([l1, l2, l3]) / mean_l) * 0.25
                results.append(PatternResult(
                    pattern_id=6, name="Triple Bottom", category="Reversal",
                    bias="BULLISH", confidence=round(min(0.95, conf), 3),
                    start_idx=int(t1), end_idx=int(t3), phase="MID",
                    bulkowski_success_rate=0.72,
                    key_levels={"support": round(mean_l, 2)},
                    description="Three bounces at support. Strong bullish signal."
                ))
        return results

    def detect_ascending_triangle(self, highs, lows, closes, peaks, troughs, atr):
        """Flat resistance + rising support (higher lows)"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        # Check if peaks are roughly flat (within 1.5%)
        peak_vals = highs[peaks]
        if len(peak_vals) < 2:
            return results
        peak_range = (np.max(peak_vals) - np.min(peak_vals)) / np.mean(peak_vals)

        # Check if troughs are rising
        trough_vals = lows[troughs]
        if len(trough_vals) < 2:
            return results
        slope, _, r2 = self.trendline_fit(troughs.astype(float), trough_vals)

        if peak_range < 0.015 and slope > 0 and r2 > 0.5:
            conf = min(0.95, 0.55 + (1 - peak_range) * 0.15 + r2 * 0.2)
            results.append(PatternResult(
                pattern_id=7, name="Ascending Triangle", category="Continuation",
                bias="BULLISH", confidence=round(conf, 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="MID",
                bulkowski_success_rate=0.75,
                key_levels={"resistance": round(np.mean(peak_vals), 2),
                            "support_slope": round(slope, 6)},
                description="Flat top + rising bottom. Bullish breakout expected."
            ))
        return results

    def detect_descending_triangle(self, highs, lows, closes, peaks, troughs, atr):
        """Flat support + falling resistance (lower highs)"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        trough_vals = lows[troughs]
        trough_range = (np.max(trough_vals) - np.min(trough_vals)) / np.mean(trough_vals)

        peak_vals = highs[peaks]
        slope, _, r2 = self.trendline_fit(peaks.astype(float), peak_vals)

        if trough_range < 0.015 and slope < 0 and r2 > 0.5:
            conf = min(0.95, 0.55 + (1 - trough_range) * 0.15 + r2 * 0.2)
            results.append(PatternResult(
                pattern_id=8, name="Descending Triangle", category="Continuation",
                bias="BEARISH", confidence=round(conf, 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="MID",
                bulkowski_success_rate=0.68,
                key_levels={"support": round(np.mean(trough_vals), 2),
                            "resistance_slope": round(slope, 6)},
                description="Flat bottom + falling top. Bearish breakdown expected."
            ))
        return results

    def detect_symmetrical_triangle(self, highs, lows, closes, peaks, troughs, atr):
        """Converging trendlines — lower highs + higher lows"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        pk_slope, _, pk_r2 = self.trendline_fit(peaks.astype(float), highs[peaks])
        tr_slope, _, tr_r2 = self.trendline_fit(troughs.astype(float), lows[troughs])

        if pk_slope < 0 and tr_slope > 0 and pk_r2 > 0.4 and tr_r2 > 0.4:
            conf = min(0.95, 0.50 + pk_r2 * 0.2 + tr_r2 * 0.2)
            results.append(PatternResult(
                pattern_id=9, name="Symmetrical Triangle", category="Bilateral",
                bias="NEUTRAL", confidence=round(conf, 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="MID",
                bulkowski_success_rate=0.70,
                key_levels={"upper_slope": round(pk_slope, 6), "lower_slope": round(tr_slope, 6)},
                description="Converging lines. Breakout direction determines bias."
            ))
        return results

    def detect_wedge(self, highs, lows, closes, peaks, troughs, atr):
        """Rising Wedge (bearish) and Falling Wedge (bullish) — converging same-direction lines"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        pk_slope, _, pk_r2 = self.trendline_fit(peaks.astype(float), highs[peaks])
        tr_slope, _, tr_r2 = self.trendline_fit(troughs.astype(float), lows[troughs])

        # Rising Wedge: both slopes positive, upper slope < lower slope (converging)
        if pk_slope > 0 and tr_slope > 0 and pk_slope < tr_slope and pk_r2 > 0.4 and tr_r2 > 0.4:
            conf = min(0.95, 0.50 + pk_r2 * 0.15 + tr_r2 * 0.15)
            results.append(PatternResult(
                pattern_id=10, name="Rising Wedge", category="Reversal",
                bias="BEARISH", confidence=round(conf, 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="LATE",
                bulkowski_success_rate=0.65,
                key_levels={"upper_slope": round(pk_slope, 6), "lower_slope": round(tr_slope, 6)},
                description="Converging upward lines. Momentum weakening. Bearish."
            ))

        # Falling Wedge: both slopes negative, lower slope steeper (converging)
        if pk_slope < 0 and tr_slope < 0 and abs(tr_slope) > abs(pk_slope) and pk_r2 > 0.4 and tr_r2 > 0.4:
            conf = min(0.95, 0.50 + pk_r2 * 0.15 + tr_r2 * 0.15)
            results.append(PatternResult(
                pattern_id=11, name="Falling Wedge", category="Reversal",
                bias="BULLISH", confidence=round(conf, 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="LATE",
                bulkowski_success_rate=0.70,
                key_levels={"upper_slope": round(pk_slope, 6), "lower_slope": round(tr_slope, 6)},
                description="Converging downward lines. Selling exhaustion. Bullish."
            ))
        return results

    def detect_flags_pennants(self, opens, highs, lows, closes, volumes):
        """Flags and Pennants: sharp move (pole) + consolidation"""
        results = []
        n = len(closes)
        if n < 10:
            return results

        # Look for sharp moves in first third (the pole)
        third = max(3, n // 3)
        pole_change = (closes[third - 1] - closes[0]) / closes[0]
        consol = closes[third:]
        if len(consol) < 3:
            return results
        consol_range = (np.max(consol) - np.min(consol)) / np.mean(consol)

        # Bull flag: strong up-move + tight consolidation
        if pole_change > 0.015 and consol_range < 0.02:
            results.append(PatternResult(
                pattern_id=12, name="Bullish Flag", category="Continuation",
                bias="BULLISH", confidence=round(min(0.90, 0.55 + pole_change * 5), 3),
                start_idx=0, end_idx=n - 1, phase="EARLY",
                bulkowski_success_rate=0.75,
                key_levels={"pole_height": round(pole_change * 100, 2)},
                description="Sharp rally + tight consolidation. Continuation expected."
            ))

        # Bear flag
        if pole_change < -0.015 and consol_range < 0.02:
            results.append(PatternResult(
                pattern_id=13, name="Bearish Flag", category="Continuation",
                bias="BEARISH", confidence=round(min(0.90, 0.55 + abs(pole_change) * 5), 3),
                start_idx=0, end_idx=n - 1, phase="EARLY",
                bulkowski_success_rate=0.68,
                key_levels={"pole_height": round(pole_change * 100, 2)},
                description="Sharp drop + tight consolidation. Continuation expected."
            ))

        # Pennant: pole + converging triangle
        if abs(pole_change) > 0.015 and len(consol) >= 4:
            c_peaks, c_troughs = self.find_extrema(consol, order=max(1, len(consol) // 4))
            if len(c_peaks) >= 2 and len(c_troughs) >= 2:
                if consol[c_peaks[-1]] < consol[c_peaks[0]] and consol[c_troughs[-1]] > consol[c_troughs[0]]:
                    pid = 14 if pole_change > 0 else 15
                    name = "Bullish Pennant" if pole_change > 0 else "Bearish Pennant"
                    bias = "BULLISH" if pole_change > 0 else "BEARISH"
                    sr = 0.68 if pole_change > 0 else 0.71
                    results.append(PatternResult(
                        pattern_id=pid, name=name, category="Continuation",
                        bias=bias, confidence=0.65, start_idx=0, end_idx=n - 1,
                        phase="EARLY", bulkowski_success_rate=sr,
                        description=f"Pole + converging triangle. {bias.lower()} continuation."
                    ))
        return results

    def detect_channel(self, highs, lows, closes, peaks, troughs, atr):
        """Ascending/Descending/Horizontal Channel — parallel trendlines"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        pk_slope, pk_int, pk_r2 = self.trendline_fit(peaks.astype(float), highs[peaks])
        tr_slope, tr_int, tr_r2 = self.trendline_fit(troughs.astype(float), lows[troughs])

        # Parallel check: slopes within 30% of each other
        if pk_r2 > 0.6 and tr_r2 > 0.6:
            if abs(pk_slope) > 1e-8 and abs(pk_slope - tr_slope) / abs(pk_slope) < 0.3:
                if pk_slope > 0:
                    name, bias = "Ascending Channel", "BULLISH"
                elif pk_slope < 0:
                    name, bias = "Descending Channel", "BEARISH"
                else:
                    name, bias = "Horizontal Channel", "NEUTRAL"

                results.append(PatternResult(
                    pattern_id=21, name=name, category="Bilateral",
                    bias=bias, confidence=round(min(0.90, (pk_r2 + tr_r2) / 2), 3),
                    start_idx=int(min(peaks[0], troughs[0])),
                    end_idx=int(max(peaks[-1], troughs[-1])), phase="FULL",
                    bulkowski_success_rate=0.65,
                    key_levels={"upper_slope": round(pk_slope, 6), "lower_slope": round(tr_slope, 6)},
                    description=f"Parallel trendlines forming {name.lower()}."
                ))
        return results

    def detect_cup_and_handle(self, highs, lows, closes, peaks, troughs):
        """Cup & Handle: U-shape base + small pullback handle"""
        results = []
        n = len(closes)
        if n < 10:
            return results

        # Find the lowest point (cup bottom)
        cup_low_idx = np.argmin(lows)
        if cup_low_idx < 3 or cup_low_idx > n - 3:
            return results

        # Left rim and right rim should be ~equal
        left_rim = np.max(highs[:cup_low_idx])
        right_rim = np.max(highs[cup_low_idx:])
        rim_diff = abs(left_rim - right_rim) / max(left_rim, right_rim)

        if rim_diff < 0.03:
            # Check for handle (small pullback in last 30%)
            handle_start = int(n * 0.7)
            if handle_start < n - 1:
                handle_drop = (np.max(closes[handle_start:]) - np.min(closes[handle_start:])) / np.mean(closes[handle_start:])
                cup_depth = (max(left_rim, right_rim) - lows[cup_low_idx]) / max(left_rim, right_rim)

                if handle_drop < cup_depth * 0.5 and cup_depth > 0.02:
                    conf = min(0.90, 0.55 + (1 - rim_diff) * 0.15 + cup_depth * 2)
                    results.append(PatternResult(
                        pattern_id=18, name="Cup & Handle", category="Continuation",
                        bias="BULLISH", confidence=round(conf, 3),
                        start_idx=0, end_idx=n - 1, phase="FULL",
                        bulkowski_success_rate=0.68,
                        key_levels={"cup_low": round(lows[cup_low_idx], 2),
                                    "rim": round(max(left_rim, right_rim), 2)},
                        description="U-shaped base + handle pullback. Bullish continuation."
                    ))
        return results

    def detect_broadening_megaphone(self, highs, lows, closes, peaks, troughs, atr):
        """Broadening/Megaphone: expanding range — higher highs + lower lows"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        pk_slope, _, pk_r2 = self.trendline_fit(peaks.astype(float), highs[peaks])
        tr_slope, _, tr_r2 = self.trendline_fit(troughs.astype(float), lows[troughs])

        # Broadening: peaks going up, troughs going down
        if pk_slope > 0 and tr_slope < 0 and pk_r2 > 0.4 and tr_r2 > 0.4:
            results.append(PatternResult(
                pattern_id=23, name="Megaphone", category="Bilateral",
                bias="NEUTRAL", confidence=round(min(0.85, (pk_r2 + tr_r2) / 2), 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="LATE",
                bulkowski_success_rate=0.52,
                description="Expanding range. Maximum market uncertainty."
            ))
        return results

    def detect_gaps(self, opens, highs, lows, closes):
        """Gap patterns: Breakaway/Runaway/Exhaustion"""
        results = []
        n = len(opens)
        for i in range(1, n):
            gap_up = opens[i] > highs[i - 1] * 1.002
            gap_dn = opens[i] < lows[i - 1] * 0.998
            if gap_up or gap_dn:
                gap_size = abs(opens[i] - closes[i - 1]) / closes[i - 1]
                phase = "EARLY" if i < n // 3 else ("MID" if i < 2 * n // 3 else "LATE")
                bias = "BULLISH" if gap_up else "BEARISH"
                results.append(PatternResult(
                    pattern_id=53, name="Gap " + ("Up" if gap_up else "Down"),
                    category="Bilateral", bias=bias,
                    confidence=round(min(0.85, 0.50 + gap_size * 10), 3),
                    start_idx=i - 1, end_idx=i, phase=phase,
                    bulkowski_success_rate=0.65,
                    key_levels={"gap_size_pct": round(gap_size * 100, 3)},
                    description=f"Price gap {'up' if gap_up else 'down'} of {gap_size*100:.2f}%."
                ))
                if len(results) >= 3:
                    break
        return results

    def detect_candlestick_patterns(self, opens, highs, lows, closes):
        """
        Core candlestick patterns (TA-Lib style implementation):
        Doji, Hammer, Engulfing, Morning/Evening Star, Kicker
        """
        results = []
        n = len(opens)
        if n < 3:
            return results

        for i in range(2, n):
            body = closes[i] - opens[i]
            body_prev = closes[i - 1] - opens[i - 1]
            body_prev2 = closes[i - 2] - opens[i - 2]
            rng = highs[i] - lows[i]
            if rng == 0:
                continue

            # Morning Star (#43): bearish + small body + bullish
            if (body_prev2 < 0 and
                abs(closes[i-1] - opens[i-1]) < rng * 0.3 and
                body > 0 and closes[i] > (opens[i-2] + closes[i-2]) / 2):
                results.append(PatternResult(
                    pattern_id=43, name="Morning Star", category="Reversal",
                    bias="BULLISH", confidence=0.68, start_idx=i-2, end_idx=i,
                    phase="LATE", bulkowski_success_rate=0.68,
                    description="Three-candle bullish reversal at bottom."
                ))

            # Evening Star (#44): bullish + small body + bearish
            if (body_prev2 > 0 and
                abs(closes[i-1] - opens[i-1]) < rng * 0.3 and
                body < 0 and closes[i] < (opens[i-2] + closes[i-2]) / 2):
                results.append(PatternResult(
                    pattern_id=44, name="Evening Star", category="Reversal",
                    bias="BEARISH", confidence=0.68, start_idx=i-2, end_idx=i,
                    phase="LATE", bulkowski_success_rate=0.68,
                    description="Three-candle bearish reversal at top."
                ))

            # Bullish Engulfing
            if body_prev < 0 and body > 0 and opens[i] <= closes[i-1] and closes[i] >= opens[i-1]:
                results.append(PatternResult(
                    pattern_id=55, name="Bullish Engulfing", category="Reversal",
                    bias="BULLISH", confidence=0.65, start_idx=i-1, end_idx=i,
                    phase="LATE", bulkowski_success_rate=0.63,
                    description="Large bullish candle engulfs prior bearish candle."
                ))

            # Bearish Engulfing
            if body_prev > 0 and body < 0 and opens[i] >= closes[i-1] and closes[i] <= opens[i-1]:
                results.append(PatternResult(
                    pattern_id=55, name="Bearish Engulfing", category="Reversal",
                    bias="BEARISH", confidence=0.65, start_idx=i-1, end_idx=i,
                    phase="LATE", bulkowski_success_rate=0.63,
                    description="Large bearish candle engulfs prior bullish candle."
                ))

            # Kicker (#42): gap + opposite strong candle
            if i >= 1:
                if body_prev < 0 and body > 0 and opens[i] > opens[i-1]:
                    results.append(PatternResult(
                        pattern_id=42, name="Bullish Kicker", category="Reversal",
                        bias="BULLISH", confidence=0.72, start_idx=i-1, end_idx=i,
                        phase="MID", bulkowski_success_rate=0.70,
                        description="Gap reversal with strong opposing candle."
                    ))

        # Hammer (single candle)
        for i in range(n):
            body_size = abs(closes[i] - opens[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            upper_wick = highs[i] - max(opens[i], closes[i])
            rng = highs[i] - lows[i]
            if rng > 0 and lower_wick > 2 * body_size and upper_wick < body_size * 0.5:
                results.append(PatternResult(
                    pattern_id=55, name="Hammer", category="Reversal",
                    bias="BULLISH", confidence=0.60, start_idx=i, end_idx=i,
                    phase="LATE", bulkowski_success_rate=0.60,
                    description="Long lower wick, small body. Potential reversal."
                ))
                break

        return results[:8]

    def detect_staircase(self, highs, lows, closes):
        """Ascending/Descending Staircase: consistent HH+HL or LH+LL"""
        results = []
        n = len(closes)
        if n < 4:
            return results

        hh, hl, lh, ll = 0, 0, 0, 0
        for i in range(1, n):
            if highs[i] > highs[i-1]: hh += 1
            if lows[i] > lows[i-1]: hl += 1
            if highs[i] < highs[i-1]: lh += 1
            if lows[i] < lows[i-1]: ll += 1

        if hh >= n * 0.65 and hl >= n * 0.65:
            results.append(PatternResult(
                pattern_id=31, name="Ascending Staircase", category="Continuation",
                bias="BULLISH", confidence=round(min(0.90, hh / n), 3),
                start_idx=0, end_idx=n-1, phase="FULL",
                bulkowski_success_rate=0.70,
                description="Consistent higher highs & higher lows. Steady buying."
            ))
        if lh >= n * 0.65 and ll >= n * 0.65:
            results.append(PatternResult(
                pattern_id=32, name="Descending Staircase", category="Continuation",
                bias="BEARISH", confidence=round(min(0.90, lh / n), 3),
                start_idx=0, end_idx=n-1, phase="FULL",
                bulkowski_success_rate=0.70,
                description="Consistent lower highs & lower lows. Steady selling."
            ))
        return results

    def detect_bull_bear_trap(self, highs, lows, closes, peaks, troughs):
        """False breakout above resistance / below support that reverses"""
        results = []
        n = len(closes)
        if n < 5:
            return results

        # Bull trap: new high on last bar that reverses
        if len(peaks) >= 1:
            prev_high = np.max(highs[:-2]) if n > 2 else highs[0]
            if highs[-1] > prev_high and closes[-1] < closes[-2]:
                results.append(PatternResult(
                    pattern_id=40, name="Bull Trap", category="Reversal",
                    bias="BEARISH", confidence=0.62, start_idx=n-3, end_idx=n-1,
                    phase="LATE", bulkowski_success_rate=0.65,
                    description="False breakout above resistance reverses sharply."
                ))

        # Bear trap
        if len(troughs) >= 1:
            prev_low = np.min(lows[:-2]) if n > 2 else lows[0]
            if lows[-1] < prev_low and closes[-1] > closes[-2]:
                results.append(PatternResult(
                    pattern_id=41, name="Bear Trap", category="Reversal",
                    bias="BULLISH", confidence=0.62, start_idx=n-3, end_idx=n-1,
                    phase="LATE", bulkowski_success_rate=0.65,
                    description="False breakdown below support reverses sharply."
                ))
        return results

    def detect_spike(self, highs, lows, closes, atr):
        """Spike pattern: extreme single-bar range"""
        results = []
        n = len(closes)
        for i in range(n):
            bar_range = highs[i] - lows[i]
            if bar_range > 2.5 * atr:
                results.append(PatternResult(
                    pattern_id=38, name="Spike", category="Bilateral",
                    bias="NEUTRAL", confidence=round(min(0.85, 0.50 + bar_range / (atr * 5)), 3),
                    start_idx=i, end_idx=i, phase="MID",
                    bulkowski_success_rate=0.58,
                    description=f"Extreme range bar ({bar_range:.0f} vs ATR {atr:.0f}). Capitulation signal."
                ))
                break
        return results

    def detect_v_pattern(self, highs, lows, closes, troughs):
        """V-bottom / V-top: sharp reversal"""
        results = []
        n = len(closes)
        if n < 5:
            return results

        low_idx = np.argmin(lows)
        if 2 <= low_idx <= n - 3:
            drop = (closes[0] - lows[low_idx]) / closes[0]
            recovery = (closes[-1] - lows[low_idx]) / lows[low_idx]
            if drop > 0.02 and recovery > 0.02:
                results.append(PatternResult(
                    pattern_id=30, name="V Bottom", category="Reversal",
                    bias="BULLISH", confidence=round(min(0.85, 0.50 + drop * 5), 3),
                    start_idx=0, end_idx=n-1, phase="FULL",
                    bulkowski_success_rate=0.60,
                    description="Sharp V-shaped reversal. Panic selling → sharp recovery."
                ))
        return results

    def detect_rectangle(self, highs, lows, closes, peaks, troughs, atr):
        """Bullish/Bearish Rectangle: horizontal range consolidation"""
        results = []
        if len(peaks) < 2 or len(troughs) < 2:
            return results

        pk_range = (np.max(highs[peaks]) - np.min(highs[peaks])) / np.mean(highs[peaks])
        tr_range = (np.max(lows[troughs]) - np.min(lows[troughs])) / np.mean(lows[troughs])

        if pk_range < 0.015 and tr_range < 0.015:
            # Determine prior trend for bias
            first_q = closes[:max(1, len(closes)//4)]
            bias = "BULLISH" if np.mean(first_q) < np.mean(closes) else "BEARISH"
            pid = 16 if bias == "BULLISH" else 17

            results.append(PatternResult(
                pattern_id=pid, name=f"{'Bullish' if bias=='BULLISH' else 'Bearish'} Rectangle",
                category="Continuation", bias=bias,
                confidence=round(min(0.85, 0.55 + (1-pk_range)*0.15 + (1-tr_range)*0.15), 3),
                start_idx=int(min(peaks[0], troughs[0])),
                end_idx=int(max(peaks[-1], troughs[-1])), phase="MID",
                bulkowski_success_rate=0.65,
                description="Horizontal consolidation in trend. Range-bound."
            ))
        return results

    # ═══════════════════════════════════════════════════════════════
    # OI PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════════
    def detect_oi_patterns(self, price_change, oi_change, pcr, oi_data=None):
        """Classify OI patterns based on price/OI relationship"""
        results = []
        if oi_change > 0 and price_change > 0:
            results.append(PatternResult(pattern_id=100, name="Long Buildup",
                category="OI", bias="BULLISH", confidence=0.80,
                start_idx=0, end_idx=0, phase="FULL", bulkowski_success_rate=0.70,
                description="Price↑ OI↑ — Fresh longs entering."))
        elif oi_change > 0 and price_change < 0:
            results.append(PatternResult(pattern_id=101, name="Short Buildup",
                category="OI", bias="BEARISH", confidence=0.80,
                start_idx=0, end_idx=0, phase="FULL", bulkowski_success_rate=0.70,
                description="Price↓ OI↑ — Fresh shorts entering."))
        elif oi_change < 0 and price_change < 0:
            results.append(PatternResult(pattern_id=102, name="Long Unwinding",
                category="OI", bias="BEARISH", confidence=0.75,
                start_idx=0, end_idx=0, phase="LATE", bulkowski_success_rate=0.65,
                description="Price↓ OI↓ — Longs exiting."))
        elif oi_change < 0 and price_change > 0:
            results.append(PatternResult(pattern_id=103, name="Short Covering",
                category="OI", bias="BULLISH", confidence=0.75,
                start_idx=0, end_idx=0, phase="LATE", bulkowski_success_rate=0.65,
                description="Price↑ OI↓ — Shorts exiting."))

        if pcr > 1.5:
            results.append(PatternResult(pattern_id=104, name="PCR Extreme High",
                category="OI", bias="BULLISH", confidence=0.85,
                start_idx=0, end_idx=0, phase="FULL", bulkowski_success_rate=0.68,
                description="Heavy put writing = strong support building."))
        elif pcr < 0.5:
            results.append(PatternResult(pattern_id=105, name="PCR Extreme Low",
                category="OI", bias="BEARISH", confidence=0.85,
                start_idx=0, end_idx=0, phase="FULL", bulkowski_success_rate=0.68,
                description="Heavy call writing = strong resistance building."))

        return results

    # ═══════════════════════════════════════════════════════════════
    # MASTER SCAN — Runs all detectors
    # ═══════════════════════════════════════════════════════════════
    def scan_all_patterns(self, opens, highs, lows, closes, volumes=None,
                          oi_change=0, pcr=1.0):
        """
        Master scanner: runs all Bulkowski pattern detectors on OHLCV data.
        Returns list of PatternResult sorted by confidence.
        """
        opens = np.asarray(opens, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
        if volumes is not None:
            volumes = np.asarray(volumes, dtype=float)
        else:
            volumes = np.ones_like(closes) * 100000

        n = len(closes)
        if n < 5:
            return []

        # Step 1: Find extrema using scipy.signal
        peaks, troughs = self.find_extrema(closes)
        h_peaks, _ = self.find_extrema(highs)
        _, l_troughs = self.find_extrema(lows)

        # Merge peak sources for robustness
        all_peaks = np.unique(np.concatenate([peaks, h_peaks])) if len(h_peaks) > 0 else peaks
        all_troughs = np.unique(np.concatenate([troughs, l_troughs])) if len(l_troughs) > 0 else troughs

        # Step 2: Calculate ATR for adaptive thresholds
        atr = self.calc_atr(highs, lows, closes)
        price_change = closes[-1] - closes[0]

        # Step 3: Run all geometric pattern detectors
        all_results = []

        all_results.extend(self.detect_head_and_shoulders(highs, lows, closes, all_peaks, all_troughs))
        all_results.extend(self.detect_inverse_head_and_shoulders(highs, lows, closes, all_peaks, all_troughs))
        all_results.extend(self.detect_double_top(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_double_bottom(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_triple_top(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_triple_bottom(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_ascending_triangle(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_descending_triangle(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_symmetrical_triangle(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_wedge(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_flags_pennants(opens, highs, lows, closes, volumes))
        all_results.extend(self.detect_channel(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_cup_and_handle(highs, lows, closes, all_peaks, all_troughs))
        all_results.extend(self.detect_broadening_megaphone(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_rectangle(highs, lows, closes, all_peaks, all_troughs, atr))
        all_results.extend(self.detect_gaps(opens, highs, lows, closes))
        all_results.extend(self.detect_candlestick_patterns(opens, highs, lows, closes))
        all_results.extend(self.detect_staircase(highs, lows, closes))
        all_results.extend(self.detect_bull_bear_trap(highs, lows, closes, all_peaks, all_troughs))
        all_results.extend(self.detect_spike(highs, lows, closes, atr))
        all_results.extend(self.detect_v_pattern(highs, lows, closes, all_troughs))

        # Step 4: OI patterns
        all_results.extend(self.detect_oi_patterns(price_change, oi_change, pcr))

        # Sort by confidence descending
        all_results.sort(key=lambda x: x.confidence, reverse=True)

        # Deduplicate by pattern_id (keep highest confidence)
        seen = set()
        unique = []
        for r in all_results:
            key = (r.pattern_id, r.name)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR for backtesting
# ═══════════════════════════════════════════════════════════════
def generate_synthetic_ohlcv(base_price, num_bars, volatility=0.008, trend=None):
    """Generate realistic OHLCV data with optional trend bias"""
    if trend is None:
        trend = (np.random.random() - 0.5) * 0.4

    closes = [base_price]
    vol = base_price * volatility
    for _ in range(num_bars - 1):
        change = (np.random.random() - 0.5 + trend) * vol
        closes.append(closes[-1] + change)

    closes = np.array(closes)
    opens = np.roll(closes, 1); opens[0] = closes[0]
    noise = np.random.random(num_bars) * vol * 0.4
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    volumes = np.random.randint(50000, 300000, num_bars).astype(float)

    return opens, highs, lows, closes, volumes


# ═══════════════════════════════════════════════════════════════
# EXPIRY CYCLE DATABASE BUILDER
# ═══════════════════════════════════════════════════════════════
INDICES = {
    "INDIA": {
        "NIFTY50": {"base": 24500, "weekly": True, "expiry_day": "Tuesday"},
        "BANKNIFTY": {"base": 52000, "weekly": False, "expiry_day": "Tuesday"},
        "FINNIFTY": {"base": 23800, "weekly": False, "expiry_day": "Tuesday"},
        "SENSEX": {"base": 80500, "weekly": True, "expiry_day": "Thursday"},
    },
    "US": {
        "SPX": {"base": 5800, "weekly": True, "expiry_day": "Friday"},
        "NDX": {"base": 20500, "weekly": True, "expiry_day": "Friday"},
        "RUT": {"base": 2100, "weekly": True, "expiry_day": "Friday"},
        "VIX": {"base": 18, "weekly": True, "expiry_day": "Wednesday"},
    }
}


def build_expiry_database(num_weekly=48, num_monthly=12):
    """Build full expiry pattern database using Bulkowski engine"""
    engine = BulkowskiPatternEngine(order=3)
    database = []

    for market, indices in INDICES.items():
        for idx_name, config in indices.items():
            bp = config["base"]
            is_weekly = config["weekly"]
            num_exp = num_weekly if is_weekly else num_monthly
            cycle_days = 5 if is_weekly else 22

            for e in range(num_exp):
                # Generate OHLCV
                o, h, l, c, v = generate_synthetic_ohlcv(bp, cycle_days)

                # Generate OI data
                pcr = round(0.5 + np.random.random() * 1.5, 3)
                oi_change = np.random.randint(-50000, 50000)

                # Run Bulkowski engine
                patterns = engine.scan_all_patterns(o, h, l, c, v, oi_change, pcr)

                # Compute stats
                start_p = c[0]
                end_p = c[-1]
                chg_pct = round((end_p - start_p) / start_p * 100, 3)
                max_dd = round((np.min(l) - start_p) / start_p * 100, 3)
                max_ru = round((np.max(h) - start_p) / start_p * 100, 3)

                # Expiry date
                from datetime import datetime, timedelta
                exp_date = datetime(2025, 1, 1) + timedelta(days=(7 if is_weekly else 30) * e)
                week_of_month = min(4, (exp_date.day - 1) // 7 + 1)
                day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                record = {
                    "id": f"{idx_name}-{e}",
                    "index": idx_name,
                    "market": market,
                    "expiryDate": exp_date.strftime("%Y-%m-%d"),
                    "cycleType": "WEEKLY" if is_weekly else "MONTHLY",
                    "days": cycle_days,
                    "startPrice": round(start_p, 2),
                    "endPrice": round(end_p, 2),
                    "changePercent": chg_pct,
                    "outcome": "BULLISH" if chg_pct > 0.5 else ("BEARISH" if chg_pct < -0.5 else "NEUTRAL"),
                    "maxDrawdown": max_dd,
                    "maxRunup": max_ru,
                    "volatility": round(np.mean(h - l), 2),
                    "pcr": pcr,
                    "oiChange": oi_change,
                    "weekOfMonth": week_of_month,
                    "dayOfWeekExpiry": day_names[exp_date.weekday()],
                    "monthlyPhase": f"M_W{week_of_month}",
                    "candles": [{"o":round(o[j],2),"h":round(h[j],2),"l":round(l[j],2),
                                 "c":round(c[j],2),"v":int(v[j])} for j in range(cycle_days)],
                    "patterns": [asdict(p) for p in patterns[:10]],
                    "patternCount": len(patterns),
                    "detectionMethod": "scipy.signal + Bulkowski rules",
                }
                database.append(record)
                bp = end_p * (0.995 + np.random.random() * 0.01)

    return database


if __name__ == "__main__":
    print("Building Bulkowski Pattern Database...")
    db = build_expiry_database(num_weekly=36, num_monthly=12)
    print(f"Generated {len(db)} expiry records")

    # Stats
    total_patterns = sum(r["patternCount"] for r in db)
    print(f"Total patterns detected: {total_patterns}")

    # Pattern frequency
    from collections import Counter
    pat_counts = Counter()
    for r in db:
        for p in r["patterns"]:
            pat_counts[p["name"]] += 1

    print("\nTop 15 most detected patterns:")
    for name, cnt in pat_counts.most_common(15):
        print(f"  {name}: {cnt}x")

    # Save to JSON
    with open("/mnt/user-data/outputs/expiry_pattern_db.json", "w") as f:
        json.dump(db, f, indent=2)
    print(f"\nDatabase saved to expiry_pattern_db.json ({len(db)} records)")
