# LLMProfileStream

A comprehensive market profile analysis pipeline with LLM-ready auction pattern classification.

## Overview

LLMProfileStream is a specialized toolkit designed for market analysis through the lens of market profile theory. It continuously ingests OHLCV (Open, High, Low, Close, Volume) data, generates multi-timeframe market profiles, and performs advanced pattern classification using a hierarchical approach optimized for LLM (Large Language Model) analysis.

Key capabilities:
- Continuous data ingestion from Google Sheets
- Multi-timeframe market profile generation (15m, 30m, 60m, 120m, 240m, daily)
- Automated value area detection and volume node (HVN/LVN) classification
- Advanced auction pattern detection and classification
- Rich metadata generation for LLM market context understanding
- Persistent storage in DuckDB and/or CSV formats

## Features

### Market Profile Analysis
- Value Area High/Low (VAH/VAL) calculation
- Volume Point of Control (VPOC) identification
- Profile shape detection (normal, P-shape, b-shape, double-distribution)
- High and Low Volume Node (HVN/LVN) detection
- Tracking of untested VPOCs across sessions
- Poor high/low detection for auction theory
- Comprehensive profile metadata (kurtosis, skewness, etc.)

### Market Pattern Detection
- Robust auction pattern classification ("Upward Value Break", "Balance Building", etc.)
- Confidence scoring for detected patterns
- Context-aware pattern notes for LLM consumption
- Open/close behavior classification relative to value areas

### Data Pipeline
- Continuous Google Sheets data ingestion
- Configurable profile parameters and timeframes
- Custom session definition (e.g., 23:00 to 22:00 next day)
- Efficient incremental data processing
- Comprehensive logging

## Installation

```bash
# Clone the repository 
git clone https://github.com/BTCJohnny/LLMProfileStream.git

# Navigate to the project directory
cd LLMProfileStream

# Run the installation script
bash scripts/install.sh
```

The installation script will:
1. Create the necessary project directories
2. Set up a Python virtual environment
3. Install required dependencies

## Configuration

Key configuration parameters are available via command-line arguments:

```bash
python ohlcv_profile_continious.py --project_root /path/to/project --hvn_count 3 --daily_start_time "23:00" --daily_stop_time "22:00"
```

Available arguments:
- `--clear`: Clear databases before running
- `--project_root`: Project root directory
- `--config_dir`: Configuration directory
- `--log_dir`: Logs directory
- `--data_dir`: Data directory for CSV output
- `--db_dir`: DuckDB database directory
- `--service_account_file`: Google service account JSON file
- `--hvn_count`: Number of High Volume Nodes to identify (default: 2)
- `--lvn_exclusion_pct`: LVN exclusion percentage (default: 0.25)
- `--max_untested_vpocs`: Maximum untested VPOCs to track (default: 5)
- `--daily_start_time`: Session start time (default: "23:00")
- `--daily_stop_time`: Session end time (default: "22:00")

## Usage

After configuration, run the main pipeline:

```bash
source venv/bin/activate
python ohlcv_profile_continious.py
```

The pipeline will:
1. Connect to Google Sheets and fetch OHLCV data
2. Process data into market profiles for all configured timeframes
3. Save results to DuckDB and/or CSV
4. Generate pattern analysis for LLM consumption
5. Run continuously, checking for new data at the configured interval

## Project Structure

```
LLMProfileStream/
├── config/                  # Configuration files including Google service account
├── data/                    # Output data directory (CSV files)
│   └── summaries/          # Pattern analysis in JSON format for LLM consumption
├── db/                      # DuckDB database files
├── helpers/                 # Helper modules
│   ├── market_profile_helpers.py  # Profile calculation and feature extraction
│   └── llm_analyzer_classifier.py # Pattern classification for LLM
├── logs/                    # Log files
├── scripts/                 # Utility scripts
│   └── install.sh           # Installation script
├── tests/                   # Test modules
├── ohlcv_profile_continious.py  # Main pipeline script
└── requirements.txt         # Python dependencies
```

## LLM Integration

The system is designed to integrate with Large Language Models by:

1. Generating structured pattern analysis with confidence scoring
2. Providing rich contextual information about market behavior
3. Outputting JSON summaries in the `data/summaries/` directory for easy consumption
4. Using a robust pattern classification system tailored for natural language understanding

## Dependencies

Main dependencies include:
- pandas
- numpy
- scipy
- duckdb
- google-api-python-client
- google-auth

See `requirements.txt` for complete list.

## License

[MIT License](LICENSE)

## Acknowledgments

- Market Profile concepts are based on the work of J. Peter Steidlmayer
- Auction Market Theory principles derive from James Dalton's "Mind Over Markets"
