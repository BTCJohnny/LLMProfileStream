"""
llm_analyzer_classifier.py

A robust classifier for market profile value area overlap and auction scenarios.
Compares the last two completed profiles and includes the opening of the current (incomplete) profile for context.
Always returns a pattern label for LLM use.
"""

def robust_auction_classifier(
    last_profile: dict,  # Most recent completed profile (for context only)
    prev_profile: dict,  # Now used as the 'current' profile for scenario logic
    prev_prev_profile: dict,  # Now used as the 'previous' profile for scenario logic
    current_open: float = None
) -> dict:
    """
    Classifies the market profile scenario using two established (completed) profiles.
    Scenario logic compares prev_profile (current) and prev_prev_profile (previous).
    Optionally includes the opening of the current (incomplete) profile for context.
    Always returns a pattern label and includes VPOCs from all three profiles for database use.
    """
    # --- Extract required fields from the 'current' (prev_profile) ---
    cur_vah = prev_profile.get('vah')
    cur_val = prev_profile.get('val')
    cur_vpoc = prev_profile.get('vpoc')
    profile_mid = prev_profile.get('profile_mid')
    skew = prev_profile.get('skewness', 0)
    kurt = prev_profile.get('kurtosis', 0)
    close_vs_value_area = prev_profile.get('close_vs_value_area')

    # --- Extract required fields from the 'previous' (prev_prev_profile) ---
    prev_vah = prev_prev_profile.get('vah')
    prev_val = prev_prev_profile.get('val')
    prev_vpoc = prev_prev_profile.get('vpoc')

    # --- Extract VPOC from the most recent completed profile for database/context ---
    last_vpoc = last_profile.get('vpoc')

    # Defensive: Ensure key current and previous data is present
    if None in [cur_vah, cur_val, cur_vpoc, profile_mid, prev_vah, prev_val, prev_vpoc]:
        return {
            'pattern_name': 'Insufficient Data',
            'pattern_confidence': 0.0,
            'pattern_notes': 'Missing value area or VPOC data.',
            'context_open': current_open,
            'vpoc_last': last_vpoc,
            'vpoc_current': cur_vpoc,
            'vpoc_previous': prev_vpoc
        }

    # --- Compute overlap ratio between current and previous value areas ---
    overlap = max(0, min(cur_vah, prev_vah) - max(cur_val, prev_val))
    va_width = cur_vah - cur_val
    prev_va_width = prev_vah - prev_val
    overlap_ratio = overlap / min(va_width, prev_va_width) if min(va_width, prev_va_width) else 0

    # --- Compute relative VPOC position for the current profile ---
    upper_third = cur_val + (2/3) * va_width
    lower_third = cur_val + (1/3) * va_width
    if cur_vpoc >= upper_third:
        vpoc_relative = 'upper_third'
    elif cur_vpoc <= lower_third:
        vpoc_relative = 'lower_third'
    else:
        vpoc_relative = 'middle_third'

    # --- Default values for classification ---
    pattern = None
    confidence = 0.7
    notes = ""

    # --- Scenario 1: Upward Value Break (clear value jump higher) ---
    if overlap_ratio < 0.25 and cur_vah > prev_vah and cur_val > prev_val:
        pattern = "Upward Value Break"
        notes = "Value area shifted up sharply with minimal overlap; indicates bullish imbalance."
        confidence = 0.85
    # --- Scenario 2: Soft Bearish Acceptance (overlap with a lower VPOC) ---
    elif overlap_ratio > 0.5 and cur_vah < prev_vah and cur_val < prev_val and vpoc_relative == 'lower_third':
        pattern = "Soft Bearish Acceptance"
        notes = "Value area overlaps lower; sellers appear to control the lower acceptance zone."
        confidence = 0.75
    # --- Scenario 3: Slow Bullish Acceptance (overlapping with VPOC in upper third) ---
    elif overlap_ratio > 0.5 and cur_vah >= prev_vah and cur_val >= prev_val and vpoc_relative == 'upper_third':
        pattern = "Slow Bullish Acceptance"
        notes = "The value area is slowly moving upward with a VPOC in the upper region; implies mild bullish bias."
        confidence = 0.7
    # --- Scenario 4: Initiating Buyers (overlap with VPOC near top) ---
    elif overlap_ratio > 0.5 and vpoc_relative == 'upper_third' and cur_vpoc > cur_vah - va_width * 0.15:
        pattern = "Initiating Buyers"
        notes = "Strong buyer activity is pushing VPOC high within the value area; potential bullish continuation."
        confidence = 0.8
    # --- Scenario 5: Buyer Fatigue Warning (VA expanding with VPOC in lower third) ---
    elif cur_vah > prev_vah and cur_val >= prev_val and vpoc_relative == 'lower_third':
        pattern = "Buyer Fatigue Warning"
        notes = "Although VA is rising, the VPOC remains low, suggesting possible buyer exhaustion."
        confidence = 0.65
    # --- Scenario 6: Volatility Expansion (value area widening sharply) ---
    elif va_width > 1.5 * prev_va_width:
        pattern = "Volatility Expansion"
        notes = "A significant widening of the value area indicates higher volatility and market indecision."
        confidence = 0.6
    # --- Scenario 7: Balance Building (current value inside previous narrow range) ---
    elif cur_vah <= prev_vah and cur_val >= prev_val:
        pattern = "Balance Building"
        notes = "The current value area is contained within the previous range; suggests a balanced auction."
        confidence = 0.7
    # --- Scenario 8: Potential Short Cover Absorption (lower overlap but with positive skew) ---
    elif overlap_ratio > 0.5 and cur_vah < prev_vah and cur_val < prev_val and skew > 0.3:
        pattern = "Potential Short Cover Absorption"
        notes = "Lower value migration with positive skewness may indicate buyers absorbing shorts; potential reversal risk."
        confidence = 0.7
    # --- Additional Scenario: Strong Bullish Momentum ---
    if not pattern and close_vs_value_area == "above_VAH":
        pattern = "Strong Bullish Momentum"
        notes = "Price closing above the value area indicates strong bullish momentum."
        confidence = 0.8
    # Fallback
    if not pattern:
        pattern = "Unclassified"
        notes = "No scenario matched."
        confidence = 0.5

    # --- Return classification and all VPOCs for database use ---
    return {
        'pattern_name': pattern,
        'pattern_confidence': confidence,
        'pattern_notes': notes,
        'context_open': current_open,
        'vpoc_last': last_vpoc,         # Most recent completed profile VPOC (for context)
        'vpoc_current': cur_vpoc,       # VPOC of the profile used as 'current' in logic
        'vpoc_previous': prev_vpoc      # VPOC of the profile used as 'previous' in logic
    } 