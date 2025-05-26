#excess_detection_helper.py

# --- Excess Detection from OHLCV Data ---

def detect_excess_from_ohlcv(profile_data: dict, minute_bars: list) -> dict:
    """
    Detect excess patterns from resampled OHLCV data using volume distribution,
    time-at-price analysis, and rejection pattern detection.
    
    Args:
        profile_data: The aggregated profile dict (from calculate_market_profile)
        minute_bars: List of 1-minute OHLCV dicts that formed this profile
        
    Returns:
        dict with excess detection results and confidence scores
    """
    if not minute_bars or not profile_data:
        return {
            'excess_high': False,
            'excess_low': False,
            'rejection_high': False,
            'rejection_low': False,
            'confidence_high': 0.0,
            'confidence_low': 0.0,
            'method_used': 'insufficient_data'
        }
    
    session_high = profile_data['high']
    session_low = profile_data['low']
    total_volume = profile_data['volume']
    
    # Method 1: Volume Distribution Analysis
    excess_high_vol, excess_low_vol = analyze_volume_distribution_excess(
        minute_bars, session_high, session_low, total_volume
    )
    
    # Method 2: Time-at-Price Analysis  
    time_excess_high, time_excess_low = analyze_time_at_extremes(
        minute_bars, session_high, session_low
    )
    
    # Method 3: Rejection Pattern Analysis
    rejection_high, rejection_low = analyze_rejection_patterns(minute_bars)
    
    # Calculate confidence scores
    confidence_high = calculate_excess_confidence(excess_high_vol, time_excess_high, rejection_high)
    confidence_low = calculate_excess_confidence(excess_low_vol, time_excess_low, rejection_low)
    
    # Final excess determination (require at least 2 of 3 methods to agree)
    excess_high_final = sum([excess_high_vol, time_excess_high, rejection_high]) >= 2
    excess_low_final = sum([excess_low_vol, time_excess_low, rejection_low]) >= 2
    
    return {
        'excess_high': excess_high_final,
        'excess_low': excess_low_final,
        'rejection_high': rejection_high,
        'rejection_low': rejection_low,
        'confidence_high': confidence_high,
        'confidence_low': confidence_low,
        'volume_excess_high': excess_high_vol,
        'volume_excess_low': excess_low_vol,
        'time_excess_high': time_excess_high,
        'time_excess_low': time_excess_low,
        'method_used': 'multi_method_consensus'
    }

def analyze_volume_distribution_excess(minute_bars: list, session_high: float, 
                                     session_low: float, total_volume: float) -> tuple:
    """
    Analyze volume distribution at price extremes to detect excess.
    Returns (excess_high_bool, excess_low_bool)
    """
    price_range = session_high - session_low
    if price_range <= 0:
        return False, False
    
    # Define extreme zones (top/bottom 2% of price range)
    high_zone_threshold = session_high - (price_range * 0.02)
    low_zone_threshold = session_low + (price_range * 0.02)
    
    volume_at_high_zone = 0
    volume_at_low_zone = 0
    
    for bar in minute_bars:
        bar_high = bar.get('high', 0)
        bar_low = bar.get('low', 0)
        bar_volume = bar.get('volume', 0)
        bar_range = bar_high - bar_low
        
        if bar_range <= 0:
            continue
            
        # Calculate volume in high zone
        if bar_high >= high_zone_threshold:
            high_zone_range = min(bar_high, session_high) - max(bar_low, high_zone_threshold)
            if high_zone_range > 0:
                volume_at_high_zone += bar_volume * (high_zone_range / bar_range)
        
        # Calculate volume in low zone
        if bar_low <= low_zone_threshold:
            low_zone_range = min(bar_high, low_zone_threshold) - max(bar_low, session_low)
            if low_zone_range > 0:
                volume_at_low_zone += bar_volume * (low_zone_range / bar_range)
    
    # Excess if less than 1.5% of total volume at extremes
    excess_high = (volume_at_high_zone / total_volume) < 0.015 if total_volume > 0 else False
    excess_low = (volume_at_low_zone / total_volume) < 0.015 if total_volume > 0 else False
    
    return excess_high, excess_low

def analyze_time_at_extremes(minute_bars: list, session_high: float, session_low: float) -> tuple:
    """
    Analyze time spent at extreme prices to detect excess.
    Returns (time_excess_high_bool, time_excess_low_bool)
    """
    if not minute_bars:
        return False, False
    
    price_range = session_high - session_low
    if price_range <= 0:
        return False, False
    
    # Define extreme zones (top/bottom 1% of price range)
    high_zone = session_high - (price_range * 0.01)
    low_zone = session_low + (price_range * 0.01)
    
    minutes_at_high = 0
    minutes_at_low = 0
    total_minutes = len(minute_bars)
    
    for bar in minute_bars:
        bar_high = bar.get('high', 0)
        bar_low = bar.get('low', 0)
        
        # Check if bar touched extreme zones
        if bar_high >= high_zone:
            minutes_at_high += 1
        if bar_low <= low_zone:
            minutes_at_low += 1
    
    # Excess if less than 5% of time spent at extremes
    time_excess_high = (minutes_at_high / total_minutes) < 0.05 if total_minutes > 0 else False
    time_excess_low = (minutes_at_low / total_minutes) < 0.05 if total_minutes > 0 else False
    
    return time_excess_high, time_excess_low

def analyze_rejection_patterns(minute_bars: list) -> tuple:
    """
    Look for sharp rejection patterns in price action at session extremes.
    Returns (rejection_high_bool, rejection_low_bool)
    """
    if len(minute_bars) < 3:
        return False, False
    
    session_high = max(bar.get('high', 0) for bar in minute_bars)
    session_low = min(bar.get('low', 0) for bar in minute_bars)
    
    # Find the bars that made the high/low
    try:
        high_bar_idx = next(i for i, bar in enumerate(minute_bars) if bar.get('high', 0) == session_high)
        low_bar_idx = next(i for i, bar in enumerate(minute_bars) if bar.get('low', 0) == session_low)
    except StopIteration:
        return False, False
    
    # Check for rejection at high
    rejection_high = False
    if high_bar_idx < len(minute_bars) - 1:  # Not the last bar
        high_bar = minute_bars[high_bar_idx]
        next_bar = minute_bars[high_bar_idx + 1]
        
        high_bar_range = high_bar.get('high', 0) - high_bar.get('low', 0)
        if high_bar_range > 0:
            close_position = (high_bar.get('close', 0) - high_bar.get('low', 0)) / high_bar_range
            
            # Strong rejection criteria
            rejection_high = (
                close_position < 0.3 and  # Closed in lower 30% of range
                next_bar.get('open', 0) < high_bar.get('close', 0) and  # Next bar opened lower
                high_bar.get('high', 0) > high_bar.get('open', 0)  # Actually made a new high
            )
    
    # Check for rejection at low (inverse logic)
    rejection_low = False
    if low_bar_idx < len(minute_bars) - 1:
        low_bar = minute_bars[low_bar_idx]
        next_bar = minute_bars[low_bar_idx + 1]
        
        low_bar_range = low_bar.get('high', 0) - low_bar.get('low', 0)
        if low_bar_range > 0:
            close_position = (low_bar.get('close', 0) - low_bar.get('low', 0)) / low_bar_range
            
            # Strong rejection criteria
            rejection_low = (
                close_position > 0.7 and  # Closed in upper 70% of range
                next_bar.get('open', 0) > low_bar.get('close', 0) and  # Next bar opened higher
                low_bar.get('low', 0) < low_bar.get('open', 0)  # Actually made a new low
            )
    
    return rejection_high, rejection_low

def calculate_excess_confidence(volume_excess: bool, time_excess: bool, rejection: bool) -> float:
    """
    Calculate confidence score for excess detection based on multiple methods.
    Returns float between 0.0 and 1.0
    """
    confidence = 0.0
    
    if volume_excess:
        confidence += 0.4  # Volume is most important indicator
    if time_excess:
        confidence += 0.3  # Time spent is secondary
    if rejection:
        confidence += 0.3  # Price action confirmation
    
    return min(1.0, confidence)

    """
    Calculate Initial Balance metrics from minute bars.
    
    Args:
        minute_bars: List of 1-minute OHLCV dicts
        ib_periods: Number of 30-minute periods for IB (default 2 = first hour)
        
    Returns:
        dict with IB high, low, range, and extension metrics
    """
    if not minute_bars:
        return {}
    
    # Calculate IB from first N 30-minute periods
    ib_minutes = ib_periods * 30
    ib_bars = minute_bars[:min(ib_minutes, len(minute_bars))]
    
    if not ib_bars:
        return {}
    
    ib_high = max(bar.get('high', 0) for bar in ib_bars)
    ib_low = min(bar.get('low', 0) for bar in ib_bars)
    ib_range = ib_high - ib_low
    
    # Session extremes
    session_high = max(bar.get('high', 0) for bar in minute_bars)
    session_low = min(bar.get('low', 0) for bar in minute_bars)
    
    # Range extensions
    upward_extension = max(0, session_high - ib_high)
    downward_extension = max(0, ib_low - session_low)
    total_extension = upward_extension + downward_extension
    
    # Extension ratio and day type classification
    extension_ratio = total_extension / ib_range if ib_range > 0 else 0
    
    if extension_ratio < 0.2:
        day_type = 'Normal'
    elif extension_ratio < 1.0:
        day_type = 'Normal Variation'
    else:
        day_type = 'Trend'
    
    return {
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_range': ib_range,
        'upward_extension': upward_extension,
        'downward_extension': downward_extension,
        'total_extension': total_extension,
        'extension_ratio': extension_ratio,
        'day_type': day_type,
        'ib_periods_used': len(ib_bars) // 30  # Actual periods captured
    }