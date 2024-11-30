"""
stock_fetcher.py

This module provides functionality to fetch and save stock data for a given stock ticker.
It retrieves historical stock data, handles potential errors, and logs the progress and outcomes.
Optimized for performance and modularity using parallel processing and efficient data handling.
"""

import os
import logging
from typing import Dict
import pandas as pd
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import get_logger  # Updated to use centralized logger

# Initialize logger
logger = get_logger('stock_fetcher', 'stock_fetcher.log')

def fetch_historical_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker and period.

    :param ticker: Stock ticker symbol.
    :param period: Period for which to fetch data (e.g., '6mo').
    :return: DataFrame containing stock data.
    """
    try:
        logger.info(f"Fetching historical data for ticker '{ticker}' with period '{period}'.")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            logger.warning(f"No historical data found for ticker '{ticker}' with period '{period}'.")
            return pd.DataFrame()

        hist.reset_index(inplace=True)
        hist['fetched_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Successfully fetched historical data for '{ticker}'.")
        return hist

    except Exception as e:
        logger.error(f"Error fetching historical data for '{ticker}': {e}")
        return pd.DataFrame()

def save_stock_data(stock_data_dict: Dict[str, pd.DataFrame], directories: Dict[str, str], logger: logging.Logger) -> Dict[str, str]:
    """
    Saves fetched stock data to CSV files.

    :param stock_data_dict: Dictionary with ticker symbols as keys and DataFrames as values.
    :param directories: Dictionary with ticker symbols as keys and directory paths as values.
    :param logger: Logger instance for logging.
    :return: Dictionary with ticker symbols as keys and file paths as values.
    """
    saved_files = {}
    for ticker, data in stock_data_dict.items():
        try:
            if data.empty:
                logger.warning(f"No stock data to save for ticker '{ticker}'.")
                continue
            file_path = os.path.join(directories[ticker], f"{ticker}_stock_data.csv")
            data.to_csv(file_path, index=False, chunksize=1000)
            logger.info(f"Stock data for '{ticker}' saved to '{file_path}'.")
            saved_files[ticker] = file_path
        except Exception as e:
            logger.error(f"Failed to save stock data for ticker '{ticker}': {e}")
    return saved_files