#!/usr/bin/env python3
"""
ohlcv_profile_continious.py

Continuously loads 1m OHLCV data from Google Sheets, generates market profiles for configurable timeframes,
and saves results to DuckDB and/or CSV. Logs all major actions and errors. Designed for modular reuse.

USAGE EXAMPLE (if you move this script to a new directory):
----------------------------------------------------------
python ohlcv_profile_continious.py \
  --project_root /absolute/path/to/your/project/root \
  --config_dir /absolute/path/to/your/project/root/config \
  --log_dir /absolute/path/to/your/project/root/logs \
  --data_dir /absolute/path/to/your/project/root/data \
  --db_dir /absolute/path/to/your/project/root/db \
  --service_account_file /absolute/path/to/your/project/root/config/your_service_account.json

You can omit any argument to use the default (relative to project root).
"""
import os
import sys
import time
import logging
import pandas as pd
import json
from datetime import datetime
import duckdb
import argparse

# === Import helper functions from the helpers directory ===
# The helpers directory contains all the supporting functions for market profile calculations and classification.
# We use relative imports to access these modules after moving the folder structure.
from helpers.market_profile_helpers import (
    calculate_market_profile, detect_profile_shape, compute_relative_volume_spike, classify_open_close_behavior,
    detect_poor_high_low, extract_key_volume_nodes, track_untested_vpocs, compute_vwap,
    label_vpoc_relative, compute_summary_features, save_profile_analysis
)

from helpers.llm_analyzer_classifier import robust_auction_classifier

# Add project root to sys.path so sibling modules (like llm_helpers) can be imported
import sys
import os

# === CONFIGURATION (edit these variables as needed, or override with CLI args) ===
# Default project root is one level up from this script
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'config')
DEFAULT_LOG_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'logs')
DEFAULT_DATA_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'data')
DEFAULT_DB_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'db')
DEFAULT_SERVICE_ACCOUNT_FILE = os.path.join(DEFAULT_CONFIG_DIR, 'river-dynamo-445508-m1-ac85f6c3600a.json')

GOOGLE_SHEET_ID = "1eejcbRqhaOzzVBGJKvoYFa3bnkfvNAEie3AhMW_LR6c"
SHEET_NAME = "ES"
SYMBOL = "ESS"
TIMEFRAMES = [15, 30, 60, 120, 240]  # in minutes, daily handled separately
SAVE_TO_CSV = True
SAVE_TO_DUCKDB = True
CHECK_INTERVAL_SECONDS = 60  # How often to check for new data (in seconds)
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# === Profile calculation parameters (now configurable) ===
HVN_COUNT = 2  # Number of HVNs to identify
LVN_EXCLUSION_PCT = 0.25  # % to exclude from top/bottom for LVNs
MAX_UNTESTED_VPOCS = 5  # Max untested VPOCs to track
LVN_MIN_SEPARATION = 3  # Minimum price levels between LVNs
LVN_CLUSTER_SIZE = 2    # Minimum price levels an LVN must span
MIN_VPOC_DISTANCE = 5.0 # Minimum distance from VPOC for HVNs
VPOC_THRESHOLD = 0.25   # Threshold for untested VPOC tracking

# === Session anchor times (now configurable) ===
DAILY_START_TIME = "23:00"  # Session starts at 23:00
DAILY_STOP_TIME = "22:00"   # Session ends at 22:00 next day

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Continuous OHLCV profile pipeline")
parser.add_argument('--clear', action='store_true', help='Clear the OHLCV and market profiles databases before running')
parser.add_argument('--project_root', type=str, default=DEFAULT_PROJECT_ROOT, help='Project root directory (default: one level up from script)')
parser.add_argument('--config_dir', type=str, default=None, help='Config directory (default: <project_root>/config)')
parser.add_argument('--log_dir', type=str, default=None, help='Logs directory (default: <project_root>/logs)')
parser.add_argument('--data_dir', type=str, default=None, help='Data directory (default: <project_root>/data)')
parser.add_argument('--db_dir', type=str, default=None, help='DB directory (default: <project_root>/db)')
parser.add_argument('--service_account_file', type=str, default=None, help='Google service account JSON file (default: <config_dir>/river-dynamo-445508-m1-ac85f6c3600a.json)')
# Add CLI args for new parameters
parser.add_argument('--hvn_count', type=int, default=HVN_COUNT, help='Number of HVNs to identify (default: 2)')
parser.add_argument('--lvn_exclusion_pct', type=float, default=LVN_EXCLUSION_PCT, help='LVN exclusion percent (default: 0.25)')
parser.add_argument('--max_untested_vpocs', type=int, default=MAX_UNTESTED_VPOCS, help='Max untested VPOCs to track (default: 10)')
parser.add_argument('--daily_start_time', type=str, default=DAILY_START_TIME, help='Session start time (default: 23:00)')
parser.add_argument('--daily_stop_time', type=str, default=DAILY_STOP_TIME, help='Session stop time (default: 22:00)')
args = parser.parse_args()

# === Resolve directories and service account file ===
PROJECT_ROOT = args.project_root
CONFIG_DIR = args.config_dir if args.config_dir else os.path.join(PROJECT_ROOT, 'config')
LOG_DIR = args.log_dir if args.log_dir else os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = args.data_dir if args.data_dir else os.path.join(PROJECT_ROOT, 'data')
DB_DIR = args.db_dir if args.db_dir else os.path.join(PROJECT_ROOT, 'db')
SERVICE_ACCOUNT_FILE = args.service_account_file if args.service_account_file else os.path.join(CONFIG_DIR, 'river-dynamo-445508-m1-ac85f6c3600a.json')

OHLCV_DB_PATH = os.path.join(DB_DIR, f"{SYMBOL}_ohlcv.duckdb")
PROFILES_DB_PATH = os.path.join(DB_DIR, f"{SYMBOL}_market_profiles.duckdb")
LOG_FILE = os.path.join(LOG_DIR, "ohlcv_profile_continious.log")

# === Ensure folders exist ===
for folder in [LOG_DIR, DATA_DIR, DB_DIR, CONFIG_DIR]:
    os.makedirs(folder, exist_ok=True)

# === Optionally clear the databases if --clear is passed ===
if args.clear:
    for db_path in [OHLCV_DB_PATH, PROFILES_DB_PATH]:
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"🧹 Cleared database due to --clear flag: {db_path}")

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ohlcv_profile_continious")

# === Google Sheets Helper ===
def get_google_sheets_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('sheets', 'v4', credentials=creds)

def get_data_from_sheets(spreadsheet_id: str, sheet_name: str = 'ES') -> pd.DataFrame:
    try:
        service = get_google_sheets_service()
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_name}!A:F"
        ).execute()
        values = result.get('values', [])
        if not values:
            logger.error("❌ No data found in the sheet.")
            return None
        df = pd.DataFrame(values[1:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp']) - pd.Timedelta(minutes=1)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"❌ Failed to read Google Sheet: {e}")
        return None

# === Data Preprocessing ===
def preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🧹 Preprocessing OHLCV data...")
    df = df.copy()
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df = df.sort_index()
    # Align all timestamps to the minute
    df.index = pd.to_datetime(df.index).floor('min')
    # Remove duplicates after flooring
    df = df[~df.index.duplicated(keep='first')]
    # Ensure continuous minute data
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min')
    df = df.reindex(full_range)
    df = df.ffill(limit=5)
    df = df.dropna()
    return df

# === Save raw OHLCV to DuckDB ===
def save_ohlcv_to_duckdb(df: pd.DataFrame, db_path: str, symbol: str):
    logger.info(f"🗃️ Saving 1m OHLCV data to DuckDB... DataFrame shape: {df.shape}")
    con = duckdb.connect(db_path)
    table_name = f"{symbol}_1m"
    # Explicitly create table if not exists
    con.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    ''')
    # Always reset index to ensure timestamp is a column
    if df.index.name != 'timestamp':
        df.index.name = 'timestamp'
    df = df.reset_index()
    # Ensure correct columns and order
    expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"⚠️ DataFrame is missing columns: {missing_cols}")
    df = df[[col for col in expected_cols if col in df.columns]]
    logger.info(f"DataFrame columns before insert: {df.columns.tolist()}")
    logger.info(f"DataFrame shape before insert: {df.shape}")

    # --- Only insert new rows ---
    # Get the latest timestamp in the table (if any)
    result = con.execute(f"SELECT MAX(timestamp) FROM {table_name}").fetchone()
    latest_db_time = result[0] if result else None

    if latest_db_time is not None:
        # Filter DataFrame to only rows with timestamp > latest_db_time
        df = df[df['timestamp'] > pd.to_datetime(latest_db_time)]
        logger.info(f"Filtered to {len(df)} new rows after latest timestamp in DB: {latest_db_time}")
    else:
        logger.info(f"No existing data in table {table_name}, inserting all rows.")

    # Insert only new rows
    if not df.empty and df.shape[1] == 6:
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        logger.info(f"Inserted {len(df)} new rows into {table_name}.")
    else:
        logger.info("No new data to insert.")

    con.close()

# === Profile Calculation Helper ===
def generate_full_profile(
    df: pd.DataFrame, tf: int,
    hvn_count, lvn_exclusion_pct, lvn_min_separation, lvn_cluster_size, min_vpoc_distance, vpoc_threshold, max_untested_vpocs,
    prev_profile: dict = None, profile_history: list = None
) -> dict:
    """
    Full-featured market profile calculation, matching DataProcessingAgent.generate_profile.
    All key parameters must be passed explicitly for clarity and to avoid ambiguity.
    """
    profile = calculate_market_profile(df, percent=0.7, tick_size=0.25)
    if not profile:
        return None

    # Add explicit timeframe tracking
    start_time = df.index[0]
    end_time = df.index[-1]
    profile.update({
        "timeframe": f"{tf}min",
        "start_time": start_time,
        "end_time": end_time,
        "missing_minutes": max(0, tf - len(df))
    })

    # Process additional metrics only if we have a valid profile
    if prev_profile:
        prev_val = prev_profile.get('val')
        prev_vah = prev_profile.get('vah')
        open_price = profile['open']
        close_price = profile['close']
        def rel_label(price):
            if prev_val is None or prev_vah is None or price is None:
                return 'unknown'
            if price > prev_vah:
                return 'Above VAH'
            elif price < prev_val:
                return 'Below VAL'
            else:
                return 'Within Value Area'
        profile['open_prev_profile'] = rel_label(open_price)
        profile['close_prev_profile'] = rel_label(close_price)
    else:
        profile['open_prev_profile'] = None
        profile['close_prev_profile'] = None

    # Calculate volume metrics
    volume_spike = compute_relative_volume_spike(df)
    profile['volume_spike_ratio'] = volume_spike

    # Process poor highs/lows
    poor_extremes = detect_poor_high_low(
        profile_distribution=profile.get('profile_distribution', {}),
        high_price=profile.get('high'),
        low_price=profile.get('low'),
        threshold=2
    )
    profile['poor_high'] = poor_extremes['poor_high']
    profile['poor_low'] = poor_extremes['poor_low']

    # Calculate volume nodes
    zone_nodes = extract_key_volume_nodes(
        profile_distribution=profile.get('profile_distribution', {}),
        hvn_count=hvn_count,
        lvn_exclusion_pct=lvn_exclusion_pct,
        lvn_min_separation=lvn_min_separation,
        lvn_cluster_size=lvn_cluster_size,
        min_vpoc_distance=min_vpoc_distance
    )
    profile['hvns'] = zone_nodes['hvns']
    profile['lvns'] = zone_nodes['lvns']

    # Track untested VPOCs
    if profile_history is not None:
        profile_history.append(profile)
        updated_profiles = track_untested_vpocs(
            profile_history,
            threshold=vpoc_threshold,
            max_untested=max_untested_vpocs
        )
        profile['untested_vpocs'] = updated_profiles[-1]['untested_vpocs']
    else:
        profile['untested_vpocs'] = []

    # Calculate VWAP
    profile['vwap'] = compute_vwap(df)

    # Add OHLCV (redundant, but for completeness)
    profile['open'] = df['open'].iloc[0]
    profile['high'] = df['high'].max()
    profile['low'] = df['low'].min()
    profile['close'] = df['close'].iloc[-1]
    profile['volume'] = df['volume'].sum()

    # Add kurtosis, skewness, and profile shape to the profile
    profile['kurtosis'] = df['close'].kurtosis()
    profile['skewness'] = df['close'].skew()
    return profile

# === Save profile to DuckDB ===
def save_profile_to_duckdb(profile: dict, db_path: str, symbol: str, tf: int, start_time, end_time):
    con = duckdb.connect(db_path)
    table_name = f"profiles_{symbol}"
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            symbol VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            timeframe VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            vah DOUBLE,
            val DOUBLE,
            mid DOUBLE,
            vpoc DOUBLE,
            kurtosis DOUBLE,
            skewness DOUBLE,
            profile_shape VARCHAR,
            vwap DOUBLE,
            profile_distribution VARCHAR
        )
    """)
    # Check for duplicate
    exists = con.execute(f"""
        SELECT COUNT(*) FROM {table_name} WHERE symbol=? AND start_time=? AND end_time=?
    """, [symbol, start_time, end_time]).fetchone()[0]
    if exists:
        con.close()
        return
    # Ensure the number of columns matches the number of values
    con.execute(f"""
        INSERT INTO {table_name} (
            symbol, start_time, end_time, timeframe, open, high, low, close, volume,
            vah, val, mid, vpoc, kurtosis, skewness, profile_shape, vwap, profile_distribution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        symbol, start_time, end_time, f"{tf}min",
        profile.get('open'), profile.get('high'), profile.get('low'), profile.get('close'), profile.get('volume'),
        profile.get('vah'), profile.get('val'), profile.get('mid'), profile.get('vpoc'),
        profile.get('kurtosis'), profile.get('skewness'), profile.get('profile_shape'),
        profile.get('vwap'), profile.get('profile_distribution')
    ])
    con.close()

# === Save profile to CSV ===
def save_profile_to_csv(profile: dict, symbol: str, tf: int, start_time, end_time):
    filename = os.path.join(DATA_DIR, f"{SHEET_NAME}_profile_{symbol}_{tf}min.csv")
    # Ensure full datetime formatting
    start_time = pd.to_datetime(start_time)
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(end_time)
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    row = {
        'symbol': symbol,
        'start_time': start_time_str,
        'end_time': end_time_str,
        'timeframe': f"{tf}min",
        'open': profile.get('open'),
        'high': profile.get('high'),
        'low': profile.get('low'),
        'close': profile.get('close'),
        'volume': profile.get('volume'),
        'vah': profile.get('vah'),
        'val': profile.get('val'),
        'mid': profile.get('mid'),
        'vpoc': profile.get('vpoc'),
        'kurtosis': profile.get('kurtosis'),
        'skewness': profile.get('skewness'),
        'profile_shape': profile.get('profile_shape'),
        'volume_spike_ratio': profile.get('volume_spike_ratio'),
        'hvns': json.dumps(profile.get('hvns', [])),
        'lvns': json.dumps(profile.get('lvns', [])),
        'untested_vpocs': json.dumps(profile.get('untested_vpocs', [])),
        'vwap': profile.get('vwap'),
        'profile_distribution': '{}',  # Only output empty dict in CSV
        'open_prev_profile': profile.get('open_prev_profile'),
        'close_prev_profile': profile.get('close_prev_profile'),
        'poor_high': profile.get('poor_high'),
        'poor_low': profile.get('poor_low'),
    }
    df_row = pd.DataFrame([row])
    header_needed = not os.path.exists(filename)
    df_row.to_csv(filename, mode='a', index=False, header=header_needed)

# === Get existing profile windows from DuckDB ===
def get_existing_profile_windows(db_path: str, symbol: str, tf: int):
    table_name = f"profiles_{symbol}"
    if not os.path.exists(db_path):
        return set()
    con = duckdb.connect(db_path)
    try:
        rows = con.execute(f"SELECT start_time, end_time FROM {table_name} WHERE timeframe=?", [f"{tf}min"]).fetchall()
        return set((str(r[0]), str(r[1])) for r in rows)
    except Exception:
        return set()
    finally:
        con.close()

# === Use CLI args for profile/session parameters ===
HVN_COUNT = args.hvn_count
LVN_EXCLUSION_PCT = args.lvn_exclusion_pct
MAX_UNTESTED_VPOCS = args.max_untested_vpocs
DAILY_START_TIME = args.daily_start_time
DAILY_STOP_TIME = args.daily_stop_time

# === Main continuous loop ===
def main_loop():
    logger.info("🚀 Starting continuous OHLCV profile pipeline...")
    # Maintain per-timeframe state
    prev_profiles = {tf: None for tf in TIMEFRAMES}
    profile_histories = {tf: [] for tf in TIMEFRAMES}
    while True:
        try:
            logger.info("📥 Loading OHLCV data from Google Sheets...")
            df = get_data_from_sheets(GOOGLE_SHEET_ID, SHEET_NAME)
            if df is None or df.empty:
                logger.warning("⚠️ No data loaded. Retrying in 1 minute.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            logger.info(f"Latest timestamp from Google Sheets: {df.index.max()}")
            df = preprocess_ohlcv(df)
            logger.info(f"Latest timestamp after preprocessing: {df.index.max()}")
            if SAVE_TO_DUCKDB:
                save_ohlcv_to_duckdb(df, OHLCV_DB_PATH, SYMBOL)
                # Log latest timestamp in DuckDB
                con = duckdb.connect(OHLCV_DB_PATH)
                latest_db_time = con.execute(f"SELECT MAX(timestamp) FROM {SYMBOL}_1m").fetchone()[0]
                logger.info(f"Latest timestamp in DuckDB: {latest_db_time}")
                con.close()

            # --- Find the first session anchor (first 23:00 in the data) ---
            # This ensures all timeframes start from 23:00, not midnight
            session_anchor_time = pd.to_datetime(DAILY_START_TIME).time()
            first_session_start = df.index[df.index.time == session_anchor_time].min()
            if pd.isnull(first_session_start):
                # If no 23:00 in data, use the first timestamp floored to 23:00
                first_date = df.index.min().date()
                first_session_start = pd.Timestamp.combine(first_date, session_anchor_time)

            # --- Process all timeframes except daily ---
            for tf in TIMEFRAMES:
                logger.info(f"⏱️ Processing {SYMBOL} {tf}m timeframe...")
                # Resample with custom origin (first 23:00)
                resampled = df.resample(f"{tf}min", origin=first_session_start)
                existing_windows = get_existing_profile_windows(PROFILES_DB_PATH, SYMBOL, tf)
                processed_windows = []
                for window_start, df_tf in resampled:
                    window_end = window_start + pd.Timedelta(minutes=tf-1)
                    # --- Only process windows that start before DAILY_STOP_TIME (e.g., 22:00) ---
                    # This prevents any profile from starting at or after the session end
                    if window_start.time() >= pd.to_datetime(DAILY_STOP_TIME).time():
                        continue
                    if df_tf.empty:
                        continue
                    key = (str(window_start.replace(second=0)), str(window_end.replace(second=0)))
                    if key in existing_windows:
                        continue
                    processed_windows.append(key)
                    # Use per-timeframe prev_profile and profile_history
                    profile = generate_full_profile(
                        df_tf, tf,
                        hvn_count=HVN_COUNT,
                        lvn_exclusion_pct=LVN_EXCLUSION_PCT,
                        lvn_min_separation=LVN_MIN_SEPARATION,
                        lvn_cluster_size=LVN_CLUSTER_SIZE,
                        min_vpoc_distance=MIN_VPOC_DISTANCE,
                        vpoc_threshold=VPOC_THRESHOLD,
                        max_untested_vpocs=MAX_UNTESTED_VPOCS,
                        prev_profile=prev_profiles[tf],
                        profile_history=profile_histories[tf]
                    )
                    if profile is None:
                        continue
                    # Update prev_profile for this timeframe
                    prev_profiles[tf] = profile
                    if SAVE_TO_DUCKDB:
                        save_profile_to_duckdb(profile, PROFILES_DB_PATH, SYMBOL, tf, window_start.replace(second=0), window_end.replace(second=0))
                        # --- LLM/Analysis summary: compute and save for this window ---
                        # Fetch all profiles for this symbol/timeframe from DB
                        con = duckdb.connect(PROFILES_DB_PATH)
                        table_name = f"profiles_{SYMBOL}"
                        query = f"SELECT * FROM {table_name} WHERE timeframe = '{tf}min' ORDER BY start_time ASC"
                        df_profiles = con.execute(query).fetchdf()
                        con.close()
                        if not df_profiles.empty:
                            profiles_list = df_profiles.to_dict(orient='records')
                            # Compute summary features for all profiles (could optimize to just last N)
                            summary_features = compute_summary_features(profiles_list)
                            # Add classifier output to every summary in the list before writing JSON
                            for i, summary in enumerate(summary_features):
                                if i < 2:
                                    # Not enough history for pattern analysis
                                    summary['pattern_name'] = 'Insufficient Data'
                                    summary['pattern_confidence'] = 0.0
                                    summary['pattern_notes'] = 'Not enough previous profiles for comparison.'
                                    summary['context_open'] = None
                                else:
                                    prev_summary = summary_features[i-1]
                                    prev_prev_summary = summary_features[i-2]
                                    # Optionally, get the opening of the next (incomplete) profile if available
                                    current_open = None
                                    if i+1 < len(summary_features):
                                        current_open = summary_features[i+1].get('open', None)
                                    # Call classifier with three established profiles
                                    result = robust_auction_classifier(summary, prev_summary, prev_prev_summary, current_open=current_open)
                                    summary['pattern_name'] = result.get('pattern_name')
                                    summary['pattern_confidence'] = result.get('pattern_confidence')
                                    summary['pattern_notes'] = result.get('pattern_notes')
                                    summary['context_open'] = result.get('context_open')
                            # Save only the last summary (for the current window) to the correct DB
                            save_profile_analysis(SYMBOL, f"{tf}min", [summary_features[-1]], db_path=PROFILES_DB_PATH)
                            # Optionally, write to JSON for LLM use
                            summary_dir = os.path.join(DATA_DIR, "summaries")
                            os.makedirs(summary_dir, exist_ok=True)
                            summary_json_path = os.path.join(summary_dir, f"{SYMBOL}_{tf}min_profile_analysis.json")
                            with open(summary_json_path, 'w') as jf:
                                json.dump(summary_features, jf, indent=2, default=str)
                    if SAVE_TO_CSV:
                        save_profile_to_csv(profile, SYMBOL, tf, window_start.replace(second=0), window_end.replace(second=0))
                if processed_windows:
                    logger.info(f"Processed {len(processed_windows)} new windows for {tf}m. First: {processed_windows[0]}, Last: {processed_windows[-1]}")
                else:
                    logger.info(f"No new windows processed for {tf}m in this run.")

            # --- Custom daily session: 23:00 to 22:00 next day ---
            logger.info(f"⏱️ Processing {SYMBOL} custom daily session ({DAILY_START_TIME} to {DAILY_STOP_TIME})...")
            # Find all session starts (all 23:00 timestamps)
            session_starts = df.index[df.index.time == session_anchor_time].unique()
            existing_windows = get_existing_profile_windows(PROFILES_DB_PATH, SYMBOL, 1440)  # Use 1440 for daily
            processed_windows = []
            for session_start in session_starts:
                # Session end is next day at 22:00 (inclusive)
                session_end = session_start + pd.Timedelta(hours=23)
                # Only process if session_end is in data
                if session_end > df.index.max():
                    continue
                key = (str(session_start.replace(second=0)), str(session_end.replace(second=0)))
                if key in existing_windows:
                    continue
                # Slice to include session_end (22:00 bar)
                df_session = df[(df.index >= session_start) & (df.index <= session_end)]
                if df_session.empty:
                    continue
                profile = generate_full_profile(
                    df_session, 1440,
                    hvn_count=HVN_COUNT,
                    lvn_exclusion_pct=LVN_EXCLUSION_PCT,
                    lvn_min_separation=LVN_MIN_SEPARATION,
                    lvn_cluster_size=LVN_CLUSTER_SIZE,
                    min_vpoc_distance=MIN_VPOC_DISTANCE,
                    vpoc_threshold=VPOC_THRESHOLD,
                    max_untested_vpocs=MAX_UNTESTED_VPOCS,
                    prev_profile=None,  # Optionally track daily prev_profile
                    profile_history=None  # Optionally track daily profile_history
                )
                if profile is None:
                    continue
                if SAVE_TO_DUCKDB:
                    save_profile_to_duckdb(profile, PROFILES_DB_PATH, SYMBOL, 1440, session_start.replace(second=0), session_end.replace(second=0))
                if SAVE_TO_CSV:
                    save_profile_to_csv(profile, SYMBOL, 1440, session_start.replace(second=0), session_end.replace(second=0))
                processed_windows.append(key)
            if processed_windows:
                logger.info(f"Processed {len(processed_windows)} new custom daily windows. First: {processed_windows[0]}, Last: {processed_windows[-1]}")
            else:
                logger.info(f"No new custom daily windows processed in this run.")

            logger.info(f"✅ Cycle complete. Sleeping {CHECK_INTERVAL_SECONDS} seconds...")
            time.sleep(CHECK_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"💥 Pipeline error: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop() 