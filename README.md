# LLMProfileStream

LLMProfileStream is a Python-based pipeline for continuous market profile analysis.  
It automatically loads 1-minute OHLCV (Open, High, Low, Close, Volume) data from Google Sheets, generates market profiles for multiple timeframes, and saves the results to DuckDB databases and/or CSV files.  
The project is designed for modular reuse, with robust logging and advanced pattern recognition features for quantitative trading, research, or automated analysis.

## Features
- Continuous data ingestion from Google Sheets
- Market profile generation for configurable timeframes
- Saves results to DuckDB and CSV
- Advanced pattern recognition and summary statistics
- Modular, reusable codebase
- Detailed logging for transparency and debugging

## Usage
See `ohlcv_profile_continious.py` for configuration and usage instructions.
