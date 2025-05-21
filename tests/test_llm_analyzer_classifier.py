import pytest
from openai_agentic_framework.llm_helpers.llm_analyzer_classifier import robust_auction_classifier

# --- Test: Scenario match (Upward Value Break) ---
def test_upward_value_break():
    # prev_prev_profile: the 'previous' profile in logic
    prev_prev = {
        'vah': 4300,
        'val': 4280,
        'vpoc': 4290,
        'profile_mid': 4290,
        'skewness': 0,
        'kurtosis': 0,
        'close_vs_value_area': 'inside_VA',
    }
    # prev_profile: the 'current' profile in logic
    prev = {
        'vah': 4320,
        'val': 4305,
        'vpoc': 4310,
        'profile_mid': 4312.5,
        'skewness': 0,
        'kurtosis': 0,
        'close_vs_value_area': 'inside_VA',
    }
    # last_profile: most recent completed (context only)
    last = {
        'vah': 4330,
        'val': 4310,
        'vpoc': 4320,
        'profile_mid': 4320,
        'skewness': 0,
        'kurtosis': 0,
        'close_vs_value_area': 'inside_VA',
    }
    result = robust_auction_classifier(last, prev, prev_prev, current_open=4335)
    # Should match Upward Value Break
    assert result['pattern_name'] == 'Upward Value Break'
    assert result['pattern_confidence'] > 0.8
    assert 'bullish' in result['pattern_notes']
    # VPOCs should be present
    assert result['vpoc_last'] == 4320
    assert result['vpoc_current'] == 4310
    assert result['vpoc_previous'] == 4290

# --- Test: Insufficient Data ---
def test_insufficient_data():
    prev_prev = {'vah': None, 'val': 4280, 'vpoc': 4290, 'profile_mid': 4290}
    prev = {'vah': 4320, 'val': 4305, 'vpoc': 4310, 'profile_mid': 4312.5}
    last = {'vah': 4330, 'val': 4310, 'vpoc': 4320, 'profile_mid': 4320}
    result = robust_auction_classifier(last, prev, prev_prev)
    assert result['pattern_name'] == 'Insufficient Data'
    assert result['pattern_confidence'] == 0.0
    assert 'Missing' in result['pattern_notes']
    # VPOCs should still be present (may be None)
    assert 'vpoc_last' in result
    assert 'vpoc_current' in result
    assert 'vpoc_previous' in result

# --- Test: Fallback Unclassified ---
def test_unclassified():
    prev_prev = {'vah': 4300, 'val': 4280, 'vpoc': 4290, 'profile_mid': 4290, 'skewness': 0, 'kurtosis': 0, 'close_vs_value_area': 'inside_VA'}
    prev = {'vah': 4301, 'val': 4281, 'vpoc': 4291, 'profile_mid': 4291, 'skewness': 0, 'kurtosis': 0, 'close_vs_value_area': 'inside_VA'}
    last = {'vah': 4302, 'val': 4282, 'vpoc': 4292, 'profile_mid': 4292, 'skewness': 0, 'kurtosis': 0, 'close_vs_value_area': 'inside_VA'}
    result = robust_auction_classifier(last, prev, prev_prev)
    assert result['pattern_name'] in ['Unclassified', 'Strong Bullish Momentum']
    # Should always return VPOCs
    assert 'vpoc_last' in result
    assert 'vpoc_current' in result
    assert 'vpoc_previous' in result

# Additional tests for other scenarios can be added as needed. 