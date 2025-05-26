# market_profile_helpers.py

import pandas as pd
import numpy as np
import scipy.stats
from typing import Optional, Dict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import json
import io
import sys
# Import the excess detection helper
from .excess_detection_helper import detect_excess_from_ohlcv

# --- Market Profile Calculation and Shape Detection ---
def calculate_market_profile(
    df_group: pd.DataFrame,
    percent: float = 0.7,
    tick_size: float = 0.25,
    recent_profiles: list = None,
    excess_detection: bool = True  # New parameter to control excess detection
) -> Optional[Dict]:
    """
    Calculate detailed ES market profile from OHLCV data
    Parameters:
      - df_group: pd.DataFrame with 'high', 'low', 'open', 'close', 'volume' columns (1-min bars)
      - percent (default 70%): volume share for Value Area
      - tick_size: minimum price increment (ES=0.25)
      - recent_profiles: list of recent profiles for HVN/LVN validation (optional)
      - excess_detection: whether to run excess detection (default True)
    Returns:
      - dict with VAH, VAL, VPOC, normalized profile, shape stats, HVNs, LVNs, excess analysis, etc.
    """
    if df_group.empty:
        return None
    high = df_group['high'].max()
    low = df_group['low'].min()
    period_open = df_group['open'].iloc[0]
    period_close = df_group['close'].iloc[-1]
    period_volume = df_group['volume'].sum()
    channel_width = tick_size
    channel_count = int((high - low) / tick_size) + 1
    bins = [0.0] * channel_count
    for _, row in df_group.iterrows():
        l = row['low']
        h = row['high']
        v = row['volume']
        start_i = max(0, int((l - low) / channel_width))
        end_i = min(channel_count - 1, int((h - low) / channel_width))
        size = max(1, end_i - start_i + 1)
        alloc = v / size
        for idx in range(start_i, end_i + 1):
            bins[idx] += alloc
    total_vol = sum(bins)
    if total_vol == 0:
        return None
    poc_vol = max(bins)
    poc_idx = bins.index(poc_vol)
    va_vol = poc_vol
    vah_i = val_i = poc_idx
    max_iterations = channel_count * 2  # Safety to prevent infinite loop
    iterations = 0
    while va_vol < percent * total_vol and (vah_i < channel_count - 1 or val_i > 0):
        if iterations > max_iterations:
            break  # Prevent infinite loop
        iterations += 1
        vol_up = bins[vah_i + 1] if vah_i < channel_count - 1 else 0
        vol_down = bins[val_i - 1] if val_i > 0 else 0
        if vol_up >= vol_down:
            vah_i += 1
            va_vol += vol_up
        else:
            val_i -= 1
            va_vol += vol_down
    get_mid = lambda i: low + (i + 0.5) * channel_width
    get_base = lambda i: low + i * channel_width
    vah = get_base(vah_i + 1)
    val = get_base(val_i)
    vpoc = get_mid(poc_idx)
    mid_price = (high + low) / 2
    raw_dist = {round(get_mid(i), 2): bins[i] for i in range(channel_count)}
    norm_dist = {price: vol / total_vol for price, vol in raw_dist.items()}
    norm_vols = np.array(list(norm_dist.values()))
    kurtosis = None
    skewness = None
    try:
        kurtosis = float(scipy.stats.kurtosis(norm_vols, bias=False)) if total_vol > 0 else 0
        skewness = float(scipy.stats.skew(norm_vols, bias=False)) if total_vol > 0 else 0
    except RuntimeWarning as w:
        import warnings
        import logging
        warnings.filterwarnings('always', category=RuntimeWarning)
        logging.warning(f"[Profile Kurtosis/Skewness Warning] {w} | start_time={df_group.index[0]}, timeframe={getattr(df_group, 'timeframe', 'unknown')}")
        kurtosis = 0
        skewness = 0
    peak_vol_ratio = max(norm_vols) if total_vol > 0 else 0
    pattern = detect_profile_shape(raw_dist)
    
    # --- NEW: LuxAlgo HVN/LVN Detection ---
    try:
        luxalgo_results = detect_hvn_lvn_peaks_and_troughs(
            profile_distribution=raw_dist,
            peak_detection_percent=0.09,
            trough_detection_percent=0.07,
            volume_threshold_percent=0.01,
            max_peaks=5,
            max_troughs=5
        )
        hvns_luxalgo = [hvn['price'] for hvn in luxalgo_results['hvns']]
        lvns_luxalgo = [lvn['price'] for lvn in luxalgo_results['lvns']]
        hvns_detailed_luxalgo = luxalgo_results['hvns']
        lvns_detailed_luxalgo = luxalgo_results['lvns']
        luxalgo_metadata = luxalgo_results['metadata']
    except Exception as e:
        print(f"⚠️ LuxAlgo HVN/LVN detection failed: {e}")
        hvns_luxalgo = []
        lvns_luxalgo = []
        hvns_detailed_luxalgo = []
        lvns_detailed_luxalgo = []
        luxalgo_metadata = {}
    
    # --- Calculate VWAP ---
    vwap = compute_vwap(df_group)
    
    result = {
        'start_time': df_group.index[0],
        'end_time': df_group.index[-1],
        'open': period_open,
        'high': high,
        'low': low,
        'close': period_close,
        'volume': period_volume,
        'vah': vah,
        'val': val,
        'vpoc': vpoc,
        'mid': mid_price,
        'profile_distribution': raw_dist,
        'normalized_distribution': norm_dist,
        'peak_volume_ratio': peak_vol_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'profile_shape': pattern,
        'tick_size': tick_size,
        'bin_count': channel_count,
        # --- LuxAlgo Method Results ---
        'hvns': hvns_luxalgo,
        'lvns': lvns_luxalgo,
        'hvns_detailed': hvns_detailed_luxalgo,
        'lvns_detailed': lvns_detailed_luxalgo,
        'hvn_lvn_metadata_luxalgo': luxalgo_metadata,
        'vwap': vwap  # Add VWAP calculation
    }

    # --- Excess Detection Integration ---
    if excess_detection:
        # Convert the DataFrame to a list of dicts for the excess detection helper
        minute_bars = df_group.to_dict('records')
        # Call the excess detection function and merge results into the profile dict
        excess_analysis = detect_excess_from_ohlcv(result, minute_bars)
        result.update(excess_analysis)
        # Now result contains all excess fields (excess_high, excess_low, etc.)

    return result

# --- NEW: LuxAlgo Peaks and Troughs Detection ---
def detect_hvn_lvn_peaks_and_troughs(
    profile_distribution: dict,
    peak_detection_percent: float = 0.09,  # 9% like LuxAlgo default
    trough_detection_percent: float = 0.07,  # 7% like LuxAlgo default  
    volume_threshold_percent: float = 0.01,  # 1% minimum volume threshold
    max_peaks: int = 5,
    max_troughs: int = 5,
    min_distance: float = 1.0  # Minimum price distance between nodes
) -> dict:
    """
    Detect HVNs (peaks) and LVNs (troughs) using LuxAlgo's proven algorithm.
    Filters out nodes that are too close together (within min_distance).
    
    Args:
        profile_distribution: Price -> volume mapping
        peak_detection_percent: % of profile rows to use for peak detection context
        trough_detection_percent: % of profile rows to use for trough detection context
        volume_threshold_percent: Minimum volume % of max to be considered
        max_peaks: Maximum HVNs to return
        max_troughs: Maximum LVNs to return
        min_distance: Minimum price distance between nodes
    """
    
    if not profile_distribution:
        return {'hvns': [], 'lvns': [], 'metadata': {}}
    
    # --- FIX: Normalize all keys to float for consistent access ---
    float_price_to_volume = {float(p): v for p, v in profile_distribution.items()}
    sorted_prices = sorted(float_price_to_volume.keys())
    volumes = [float_price_to_volume[p] for p in sorted_prices]
    # --- END FIX ---
    
    if not volumes:
        return {'hvns': [], 'lvns': [], 'metadata': {}}
    
    max_volume = max(volumes)
    volume_threshold = max_volume * volume_threshold_percent
    profile_rows = len(volumes)
    
    # Calculate detection windows (like PineScript's peaksNumberOfNodes)
    peak_window = max(1, int(profile_rows * peak_detection_percent))
    trough_window = max(1, int(profile_rows * trough_detection_percent))
    
    hvns = []  # Peaks
    lvns = []  # Troughs
    
    # === HVN (Peak) Detection ===
    for i in range(peak_window, profile_rows - peak_window):
        current_volume = volumes[i]
        current_price = sorted_prices[i]
        
        # Skip if below volume threshold
        if current_volume < volume_threshold:
            continue
            
        # Check if it's a peak (higher than all neighbors in window)
        is_peak = True
        
        # Check preceding nodes
        for j in range(i - peak_window, i):
            if current_volume <= volumes[j]:
                is_peak = False
                break
        
        if not is_peak:
            continue
            
        # Check succeeding nodes  
        for j in range(i + 1, i + peak_window + 1):
            if current_volume <= volumes[j]:
                is_peak = False
                break
        
        if is_peak:
            hvns.append({
                'price': current_price,
                'volume': current_volume,
                'confidence': current_volume / max_volume,  # Relative strength
                'window_size': peak_window,
                'node_type': 'peak',
                'detection_method': 'luxalgo_peaks'
            })
    
    # === LVN (Trough) Detection ===
    for i in range(trough_window, profile_rows - trough_window):
        current_volume = volumes[i]
        current_price = sorted_prices[i]
        
        # For troughs, we want low volume relative to max
        if current_volume > max_volume * 0.5:  # Skip if too high volume
            continue
            
        # Check if it's a trough (lower than all neighbors in window)
        is_trough = True
        
        # Check preceding nodes
        for j in range(i - trough_window, i):
            if current_volume >= volumes[j]:
                is_trough = False
                break
        
        if not is_trough:
            continue
            
        # Check succeeding nodes
        for j in range(i + 1, i + trough_window + 1):
            if current_volume >= volumes[j]:
                is_trough = False
                break
        
        if is_trough:
            lvns.append({
                'price': current_price,
                'volume': current_volume,
                'confidence': 1.0 - (current_volume / max_volume),  # Lower volume = higher confidence for LVN
                'window_size': trough_window,
                'node_type': 'trough',
                'detection_method': 'luxalgo_troughs'
            })
    
    # Sort and limit results
    hvns.sort(key=lambda x: x['confidence'], reverse=True)
    lvns.sort(key=lambda x: x['confidence'], reverse=True)
    
    # After sorting and before limiting, filter by min_distance
    def filter_by_min_distance(nodes, min_distance):
        filtered = []
        for node in nodes:
            if all(abs(node['price'] - f['price']) >= min_distance for f in filtered):
                filtered.append(node)
        return filtered
    hvns = filter_by_min_distance(hvns, min_distance)
    lvns = filter_by_min_distance(lvns, min_distance)
    # Now limit to max_peaks/max_troughs
    hvns = hvns[:max_peaks]
    lvns = lvns[:max_troughs]
    
    return {
        'hvns': hvns,
        'lvns': lvns,
        'metadata': {
            'peak_window': peak_window,
            'trough_window': trough_window,
            'volume_threshold': volume_threshold,
            'max_volume': max_volume,
            'total_price_levels': profile_rows,
            'detection_method': 'luxalgo_peaks_troughs'
        }
    }

def detect_profile_shape(raw_dist: dict) -> str:
    """
    Determine the market profile shape based on the raw distribution dictionary.
    Expects raw_dist to be a dictionary with keys as prices (float or string convertible to float)
    and values as volume amounts.
    Returns a string describing the profile shape.
    """
    if not isinstance(raw_dist, dict) or len(raw_dist) < 3:
        return "unknown"
    prices = np.array([float(price) for price in raw_dist.keys()])
    vols = np.array(list(raw_dist.values()))
    if np.sum(vols) == 0:
        return "empty"
    vols_norm = vols / (np.max(vols) + 1e-9)
    smoothed = gaussian_filter1d(vols_norm, sigma=1)
    peaks, props = find_peaks(smoothed, prominence=0.15)
    num_strong_peaks = len(peaks)
    skew = None
    if num_strong_peaks >= 1:
        max_idx = peaks[np.argmax(smoothed[peaks])]
        skew = (np.mean(prices) - prices[max_idx]) / (np.ptp(prices) + 1e-9)
    pattern = "other"
    if num_strong_peaks == 1:
        if skew is not None:
            if skew > 0.15:
                pattern = "b-shape"
            elif skew < -0.15:
                pattern = "P-shape"
            else:
                pattern = "normal"
    elif num_strong_peaks >= 2:
        pattern = "double-distro"
    else:
        start_vol = smoothed[0]
        end_vol = smoothed[-1]
        if (start_vol > 0.5) or (end_vol > 0.5):
            pattern = "trend"
        else:
            pattern = "other"
    return pattern

# --- Relative Volume Spike ---
def compute_relative_volume_spike(df, method="mean"):
    """
    Compute relative volume spike factor for the latest bar.
    Parameters:
    - df (pd.DataFrame): OHLCV data (timestamp-indexed)
    - method (str): 'mean' or 'median' to use as baseline
    Returns:
    - float: Spike factor (e.g., 2.5 means 2.5x average volume)
    """
    if len(df) < 2:
        return None
    current_volume = df.iloc[-1]['volume']
    historical = df.iloc[:-1]['volume']
    if method == "mean":
        baseline = historical.mean()
    elif method == "median":
        baseline = historical.median()
    else:
        raise ValueError("method must be 'mean' or 'median'")
    if baseline == 0:
        return None
    return round(current_volume / baseline, 2)

# --- Open/Close Position Classification ---
def classify_open_close_behavior(curr_profile, prev_profile):
    """
    Determine open/close position vs. previous value area and interpret behavior.
    Inputs:
      - curr_profile: dict with current profile values (open, close, VAH, VAL)
      - prev_profile: dict with previous profile values (VAH, VAL)
    Outputs:
      - Tuple of strings: (open_location, close_location, behavioral_insight)
    """
    if curr_profile['open'] > prev_profile['vah']:
        open_loc = 'Above Previous VAH'
    elif curr_profile['open'] < prev_profile['val']:
        open_loc = 'Below Previous VAL'
    else:
        open_loc = 'Within Previous Value'
    if curr_profile['close'] > curr_profile['vah']:
        close_loc = 'Above VAH'
    elif curr_profile['close'] < curr_profile['val']:
        close_loc = 'Below VAL'
    else:
        close_loc = 'Within Value'
    if open_loc == 'Below Previous VAL' and close_loc == 'Above VAH':
        insight = 'Strong reversal signal: opened below value, closed above VAH.'
    elif open_loc == 'Above Previous VAH' and close_loc == 'Below VAL':
        insight = 'Exhaustion pattern: opened above value, closed below VAL.'
    elif open_loc == 'Within Previous Value' and close_loc == 'Within Value':
        insight = 'Balanced session with potential continuation.'
    elif open_loc == 'Above Previous VAH' and close_loc == 'Above VAH':
        insight = 'Sustained positioning: both open and close above previous VAH.'
    elif open_loc == 'Below Previous VAL' and close_loc == 'Below VAL':
        insight = 'Sustained positioning: both open and close below previous VAL.'
    elif open_loc == 'Within Previous Value':
        if close_loc == 'Above VAH':
            insight = 'Bullish breakout from value area.'
        elif close_loc == 'Below VAL':
            insight = 'Bearish breakdown from value area.'
        else:
            insight = 'Moderate transition within profile range.'
    elif close_loc == 'Within Value':
        if open_loc == 'Above Previous VAH':
            insight = 'Reversion to value from above.'
        elif open_loc == 'Below Previous VAL':
            insight = 'Reversion to value from below.'
        else:
            insight = 'Moderate transition within profile range.'
    else:
        insight = 'Moderate transition within profile range.'
    return open_loc, close_loc, insight

# --- Poor High/Low Detection ---
def detect_poor_high_low(profile_distribution, high_price, low_price, threshold=2):
    """
    Detect whether the session ends in a poor high or poor low.
    Parameters:
    - profile_distribution (dict): price -> volume mapping
    - high_price (float): session high
    - low_price (float): session low
    - threshold (float): volume threshold to consider it "poor"
    Returns:
    - dict: {'poor_high': bool, 'poor_low': bool}
    """
    if not profile_distribution:
        return {'poor_high': None, 'poor_low': None}
    sorted_prices = sorted(profile_distribution.keys())
    top_price = sorted_prices[-1]
    bottom_price = sorted_prices[0]
    poor_high = (abs(top_price - high_price) < 1e-6 and profile_distribution[top_price] <= threshold)
    poor_low = (abs(bottom_price - low_price) < 1e-6 and profile_distribution[bottom_price] <= threshold)
    return {
        'poor_high': poor_high,
        'poor_low': poor_low
    }

# --- Untested VPOC Tracker ---
def track_untested_vpocs(profiles, threshold=0.0, max_untested=3):
    """
    Track and attach a rolling list of untested VPOCs to each profile.
    Parameters:
    - profiles (list): list of profile dicts in chronological order
    - threshold (float): optional tolerance for test proximity
    - max_untested (int): maximum number of untested VPOCs to keep
    Returns:
    - list: same profile list, each with 'untested_vpocs' field added
    """
    untested = []
    for profile in profiles:
        current_low = profile.get("low")
        current_high = profile.get("high")
        current_vpoc = profile.get("vpoc")
        tested = [vpoc for vpoc in untested if current_low - threshold <= vpoc <= current_high + threshold]
        untested = [vpoc for vpoc in untested if vpoc not in tested]
        if current_vpoc is not None:
            untested.append(current_vpoc)
        untested = untested[-5:]
        profile["untested_vpocs"] = untested.copy()
    return profiles

# --- VWAP Calculation ---
def compute_vwap(df: pd.DataFrame) -> float:
    """
    Compute the Volume-Weighted Average Price (VWAP) for a given OHLCV DataFrame.
    Parameters:
    - df (pd.DataFrame): must include 'high', 'low', 'close', and 'volume'
    Returns:
    - float: final VWAP value
    """
    if df.empty or not {'high', 'low', 'close', 'volume'}.issubset(df.columns):
        return None
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
    return round(vwap, 4) if pd.notnull(vwap) else None

def label_vpoc_relative(profile: dict) -> str:
    """
    Classifies the VPOC location relative to VAL/VAH into one of:
    'lower_third', 'middle_third', 'upper_third', or 'unknown/center' if there's an edge case.
    """
    try:
        val = profile['val']
        vah = profile['vah']
        vpoc = profile['vpoc']
        width = vah - val
        if abs(width) < 1e-9:
            return "center"  # Avoid division by zero
        pos = (vpoc - val) / width
        if pos <= 1/3:
            return "lower_third"
        elif pos >= 2/3:
            return "upper_third"
        else:
            return "middle_third"
    except Exception as e:
        # Logging or debugging can be added here
        return "unknown"

def compute_summary_features(profiles: list) -> list:
    """
    Given a list of profiles (entire dataset if desired),
    computes a variety of summary metrics for each profile in chronological order.
    This includes:
    - Value area migration (gap_higher, overlapping_higher, etc.)
    - Overlap ratio
    - Range expansion/contraction
    - VPOC location (lower_third, middle_third, upper_third)
    - Skewness/kurtosis (if included in profile data from the DB)
    - Profile shape (if included in profile data from the DB)
    """
    output = []
    for i, cur in enumerate(profiles):
        # Provide safe fallback if any fields are missing
        val = cur.get('val', None)
        vah = cur.get('vah', None)
        vpoc = cur.get('vpoc', None)
        close_price = cur.get('close', None)

        # Identify previous profile to detect trends
        if i == 0:
            prev = cur
            value_trend = "N/A"
            overlap_perc = "N/A"
            range_change = "N/A"
            open_loc, close_loc, insight = "N/A", "N/A", "N/A"
        else:
            prev = profiles[i - 1]
            # Value area migration
            if val is not None and vah is not None and prev.get('val') is not None and prev.get('vah') is not None:
                if val > prev['vah']:
                    value_trend = "gap_higher"
                elif vah < prev['val']:
                    value_trend = "gap_lower"
                elif val >= prev['val'] and vah > prev['vah']:
                    value_trend = "overlapping_higher"
                elif vah <= prev['vah'] and val < prev['val']:
                    value_trend = "overlapping_lower"
                else:
                    value_trend = "sideways"
            else:
                value_trend = "unknown"

            # Overlap %
            if value_trend in ["gap_higher", "gap_lower", "unknown"]:
                overlap_perc = 0
            else:
                overlap = max(0, min(vah, prev['vah']) - max(val, prev['val']))
                prev_width = prev['vah'] - prev['val']
                cur_width = vah - val
                overlap_perc = overlap / min(prev_width, cur_width) if min(prev_width, cur_width) else 0

            # Range width change
            if value_trend in ["gap_higher", "gap_lower", "unknown"]:
                range_change = "N/A"
            else:
                delta = (vah - val) - (prev['vah'] - prev['val'])
                if abs(delta) < 1e-4:
                    range_change = "stable"
                elif delta > 0:
                    range_change = "expanding"
                else:
                    range_change = "contracting"

            # Use classify_open_close_behavior for open/close position and insight
            open_loc, close_loc, insight = classify_open_close_behavior(cur, prev)

        # VPOC relative
        vpoc_loc = label_vpoc_relative(cur)

        # Build summary row
        output.append({
            "timestamp": cur.get("start_time", ""),
            "timeframe": cur.get("timeframe", ""),
            "vah": vah,
            "val": val,
            "vpoc": vpoc,
            "profile_high": cur.get("high", None),
            "profile_low": cur.get("low", None),
            "close": close_price,
            "profile_mid": (vah + val) / 2 if (vah is not None and val is not None) else None,
            "vah_val_width": (vah - val) if (vah is not None and val is not None) else None,
            "volume": cur.get("volume", None),
            # If the DB includes skewness/kurtosis, capture them:
            "skewness": cur.get("skewness", 0),
            "kurtosis": cur.get("kurtosis", 0),
            "profile_shape": cur.get("profile_shape", "unknown"),
            "value_area_trend": value_trend,
            "value_area_overlap_ratio": round(overlap_perc, 2) if overlap_perc != "N/A" else "N/A",
            "value_area_width_change": range_change,
            "vpoc_relative_location": vpoc_loc,
            "open_vs_prev_value_area": open_loc,
            "close_vs_value_area": close_loc,
            "pattern_open_close_behavior": insight,
            # New fields:
            "open": cur.get("open", None),
            "hvns": json.dumps(cur.get("hvns", [])),
            "lvns": json.dumps(cur.get("lvns", [])),
            "untested_vpocs": json.dumps(cur.get("untested_vpocs", [])),
            "vwap": cur.get("vwap", None),
            "mid": cur.get("mid", None),
            # Add previous value area and VPOC for classifier use
            "previous_vah": prev.get("vah", None),
            "previous_val": prev.get("val", None),
            "previous_vpoc": prev.get("vpoc", None),
            # --- Excess detection fields ---
            "excess_high": cur.get("excess_high", None),
            "excess_low": cur.get("excess_low", None),
            "confidence_high": cur.get("confidence_high", None),
            "confidence_low": cur.get("confidence_low", None),
            "rejection_high": cur.get("rejection_high", None),
            "rejection_low": cur.get("rejection_low", None),
            "volume_excess_high": cur.get("volume_excess_high", None),
            "volume_excess_low": cur.get("volume_excess_low", None),
            "time_excess_high": cur.get("time_excess_high", None),
            "time_excess_low": cur.get("time_excess_low", None)
        })
    return output

def save_profile_analysis(symbol: str, timeframe: str, features: list, db_path: str = None) -> None:
    """
    Save all processed profile analyses into a DuckDB file.
    Creates or updates the symbol-specific table if necessary.
    Table name format: profile_analysis_{symbol} (e.g., profile_analysis_ESS)
    The table schema includes all relevant summary/labeling fields for LLM and analysis.
    """
    import os
    import duckdb
    # Use a default DB path if not provided
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if db_path is None:
        db_path = os.path.join(ROOT_DIR, "profiles_summary.duckdb")

    # Create the table if it does not exist - include symbol in table name
    table_name = f"profile_analysis_{symbol}"
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        symbol TEXT,
        timeframe TEXT,
        timestamp TIMESTAMP,
        vah DOUBLE,
        val DOUBLE,
        vpoc DOUBLE,
        profile_high DOUBLE,
        profile_low DOUBLE,
        close DOUBLE,
        profile_mid DOUBLE,
        vah_val_width DOUBLE,
        volume DOUBLE,
        skewness DOUBLE,
        kurtosis DOUBLE,
        profile_shape TEXT,
        value_area_trend TEXT,
        value_area_overlap_ratio TEXT,
        value_area_width_change TEXT,
        vpoc_relative_location TEXT,
        open_vs_prev_value_area TEXT,
        close_vs_value_area TEXT,
        pattern_open_close_behavior TEXT,
        open DOUBLE,
        hvns TEXT,
        lvns TEXT,
        untested_vpocs TEXT,
        vwap DOUBLE,
        mid DOUBLE,
        previous_vah DOUBLE,
        previous_val DOUBLE,
        previous_vpoc DOUBLE,
        -- Excess detection fields
        excess_high BOOLEAN,
        excess_low BOOLEAN,
        confidence_high DOUBLE,
        confidence_low DOUBLE,
        rejection_high BOOLEAN,
        rejection_low BOOLEAN,
        volume_excess_high BOOLEAN,
        volume_excess_low BOOLEAN,
        time_excess_high BOOLEAN,
        time_excess_low BOOLEAN
    )
    """

    with duckdb.connect(database=db_path) as con:
        con.execute(create_table_query)
        for feature in features:
            # Extract each field from feature (with defaults if key is missing).
            row_symbol = symbol
            row_timeframe = timeframe
            row_timestamp = feature.get("timestamp", None)
            row_vah = feature.get("vah", None)
            row_val = feature.get("val", None)
            row_vpoc = feature.get("vpoc", None)
            row_profile_high = feature.get("profile_high", None)
            row_profile_low = feature.get("profile_low", None)
            row_close = feature.get("close", None)
            row_profile_mid = feature.get("profile_mid", None)
            row_vah_val_width = feature.get("vah_val_width", None)
            row_volume = feature.get("volume", None)
            row_skewness = feature.get("skewness", 0)
            row_kurtosis = feature.get("kurtosis", 0)
            row_profile_shape = feature.get("profile_shape", "unknown")
            row_value_area_trend = feature.get("value_area_trend", "")
            val_overlap = feature.get("value_area_overlap_ratio", "N/A")
            row_value_area_overlap_ratio = str(val_overlap)
            row_value_area_width_change = feature.get("value_area_width_change", "")
            row_vpoc_relative_location = feature.get("vpoc_relative_location", "")
            row_open_vs_prev_value_area = feature.get("open_vs_prev_value_area", "")
            row_close_vs_value_area = feature.get("close_vs_value_area", "")
            row_pattern_open_close_behavior = feature.get("pattern_open_close_behavior", "")
            row_open = feature.get("open", None)
            row_hvns = feature.get("hvns", "[]")
            row_lvns = feature.get("lvns", "[]")
            row_untested_vpocs = feature.get("untested_vpocs", "[]")
            row_vwap = feature.get("vwap", None)
            row_mid = feature.get("mid", None)
            row_previous_vah = feature.get("previous_vah", None)
            row_previous_val = feature.get("previous_val", None)
            row_previous_vpoc = feature.get("previous_vpoc", None)
            # --- Excess detection fields ---
            row_excess_high = feature.get("excess_high", None)
            row_excess_low = feature.get("excess_low", None)
            row_confidence_high = feature.get("confidence_high", None)
            row_confidence_low = feature.get("confidence_low", None)
            row_rejection_high = feature.get("rejection_high", None)
            row_rejection_low = feature.get("rejection_low", None)
            row_volume_excess_high = feature.get("volume_excess_high", None)
            row_volume_excess_low = feature.get("volume_excess_low", None)
            row_time_excess_high = feature.get("time_excess_high", None)
            row_time_excess_low = feature.get("time_excess_low", None)

            # Remove any existing record with the same symbol, timeframe, and timestamp
            delete_query = f"""
            DELETE FROM {table_name}
            WHERE symbol = ?
              AND timeframe = ?
              AND timestamp = ?
            """
            con.execute(delete_query, (row_symbol, row_timeframe, row_timestamp))

            # Insert the new record
            insert_query = f"""
            INSERT INTO {table_name} (
                symbol,
                timeframe,
                timestamp,
                vah,
                val,
                vpoc,
                profile_high,
                profile_low,
                close,
                profile_mid,
                vah_val_width,
                volume,
                skewness,
                kurtosis,
                profile_shape,
                value_area_trend,
                value_area_overlap_ratio,
                value_area_width_change,
                vpoc_relative_location,
                open_vs_prev_value_area,
                close_vs_value_area,
                pattern_open_close_behavior,
                open,
                hvns,
                lvns,
                untested_vpocs,
                vwap,
                mid,
                previous_vah,
                previous_val,
                previous_vpoc,
                excess_high,
                excess_low,
                confidence_high,
                confidence_low,
                rejection_high,
                rejection_low,
                volume_excess_high,
                volume_excess_low,
                time_excess_high,
                time_excess_low
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """
            con.execute(insert_query, (
                row_symbol,
                row_timeframe,
                row_timestamp,
                row_vah,
                row_val,
                row_vpoc,
                row_profile_high,
                row_profile_low,
                row_close,
                row_profile_mid,
                row_vah_val_width,
                row_volume,
                float(row_skewness) if row_skewness is not None else None,
                float(row_kurtosis) if row_kurtosis is not None else None,
                row_profile_shape,
                row_value_area_trend,
                row_value_area_overlap_ratio,
                row_value_area_width_change,
                row_vpoc_relative_location,
                row_open_vs_prev_value_area,
                row_close_vs_value_area,
                row_pattern_open_close_behavior,
                row_open,
                row_hvns,
                row_lvns,
                row_untested_vpocs,
                row_vwap,
                row_mid,
                row_previous_vah,
                row_previous_val,
                row_previous_vpoc,
                row_excess_high,
                row_excess_low,
                row_confidence_high,
                row_confidence_low,
                row_rejection_high,
                row_rejection_low,
                row_volume_excess_high,
                row_volume_excess_low,
                row_time_excess_high,
                row_time_excess_low
            ))

def to_py_bool(val):
    """Convert numpy.bool_ or pandas boolean to native Python bool (or None)."""
    import numpy as np
    if val is None or isinstance(val, bool):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    return bool(val)

def build_summary_feature(profile: dict, prev_profile: dict | None = None, *, timeframe: str = "") -> dict:
    """Build a summary-feature dict that matches the database schema.

    Parameters
    ----------
    profile : dict
        The current profile produced by `generate_full_profile`.
    prev_profile : dict | None, optional
        Previous profile for context, by default None.
    timeframe : str, optional
        Timeframe label (e.g. "15min"), by default "".

    Returns
    -------
    dict
        A dict whose keys exactly match the columns in `save_profile_analysis`.
    """
    # Helper safe-get
    g = profile.get
    prev = prev_profile or {}

    summary = {
        "timestamp": g("start_time"),
        "timeframe": timeframe,
        "vah": g("vah"),
        "val": g("val"),
        "vpoc": g("vpoc"),
        "profile_high": g("high"),
        "profile_low": g("low"),
        "close": g("close"),
        "profile_mid": (g("vah") + g("val")) / 2 if (g("vah") is not None and g("val") is not None) else None,
        "vah_val_width": (g("vah") - g("val")) if (g("vah") is not None and g("val") is not None) else None,
        "volume": g("volume"),
        "skewness": g("skewness"),
        "kurtosis": g("kurtosis"),
        "profile_shape": g("profile_shape"),
        "value_area_trend": None,
        "value_area_overlap_ratio": None,
        "value_area_width_change": None,
        "vpoc_relative_location": None,
        "open_vs_prev_value_area": g("open_prev_profile"),
        "close_vs_value_area": None,
        "pattern_open_close_behavior": None,
        "open": g("open"),
        "hvns": json.dumps(g("hvns", [])),
        "lvns": json.dumps(g("lvns", [])),
        "untested_vpocs": json.dumps(g("untested_vpocs", [])),
        "vwap": g("vwap"),
        "mid": g("mid"),
        "previous_vah": prev.get("vah"),
        "previous_val": prev.get("val"),
        "previous_vpoc": prev.get("vpoc"),
        "excess_high": to_py_bool(g("excess_high")),
        "excess_low": to_py_bool(g("excess_low")),
        "confidence_high": g("confidence_high"),
        "confidence_low": g("confidence_low"),
        "rejection_high": to_py_bool(g("rejection_high")),
        "rejection_low": to_py_bool(g("rejection_low")),
        "volume_excess_high": to_py_bool(g("volume_excess_high")),
        "volume_excess_low": to_py_bool(g("volume_excess_low")),
        "time_excess_high": to_py_bool(g("time_excess_high")),
        "time_excess_low": to_py_bool(g("time_excess_low")),
        # LLM pattern placeholder
        "pattern_name": "Current Profile",
        "pattern_confidence": 1.0,
        "pattern_notes": "Live profile data",
        "context_open": None,
    }
    return summary

def save_summary_to_csv(symbol: str, timeframe: str, summary: dict, data_dir: str):
    """Append a summary-feature row to a CSV, creating it with headers if needed."""
    import pandas as pd
    import os
    filename = os.path.join(data_dir, f"{symbol}_{timeframe}_profile_analysis.csv")
    df_row = pd.DataFrame([summary])
    header_needed = not os.path.exists(filename)
    df_row.to_csv(filename, mode="a", index=False, header=header_needed)

def append_summary_to_json(symbol: str, timeframe: str, summary: dict, data_dir: str):
    """Append a summary-feature dict to a JSON file containing a list of summaries."""
    import os, json
    path = os.path.join(data_dir, f"{symbol}_{timeframe}_profile_analysis.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as jf:
                existing = json.load(jf)
                if not isinstance(existing, list):
                    existing = [existing]
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(summary)
    with open(path, "w") as jf:
        json.dump(existing, jf, indent=2, default=str)