"""
options_fetcher.py

This module provides functionality to fetch and save options data for a given stock ticker within a specified date range.
It retrieves expiration dates, fetches calls and puts data for each expiration date in parallel, streams large data chunks to reduce memory usage,
and saves them as separate CSV files. Robust error handling and logging are implemented to ensure reliability and ease of debugging.
"""

import os
import json  # Add this import at the top of the file
from typing import List
import pandas as pd
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from colorama import Fore, Style

from logger import get_logger  # Updated to use centralized logger

# Initialize logger
logger = get_logger('options_fetcher', 'logs/options_fetcher.log')

def get_expiration_dates(ticker: str, start_date: str, end_date: str) -> List[str]:
    """
    Retrieves a list of expiration dates for the given ticker within the specified date range.

    :param ticker: Stock ticker symbol.
    :param start_date: Start date for the 6-month period (format: 'YYYY-MM-DD').
    :param end_date: End date for the 6-month period (format: 'YYYY-MM-DD').
    :return: List of expiration dates as strings.
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            logger.warning(f"No expiration dates found for ticker '{ticker}'.")
            return []

        # Convert expiration dates to datetime objects
        expirations_dt = []
        for date_str in expirations:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                expirations_dt.append(date_obj)
            except ValueError as ve:
                logger.error(f"Invalid date format '{date_str}' for ticker '{ticker}': {ve}")
                continue  # Skip invalid date formats

        # Define start and end datetime objects
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as ve:
            logger.error(f"Invalid start or end date format: {ve}")
            return []

        # Filter expiration dates within the specified range
        filtered_expirations = [
            date.strftime('%Y-%m-%d') for date in expirations_dt
            if start_dt <= date <= end_dt
        ]

        if not filtered_expirations:
            logger.warning(f"No expiration dates within the period {start_date} to {end_date} for ticker '{ticker}'.")

        return filtered_expirations

    except yf.shared._exceptions.YFError as yf_err:
        logger.error(f"Yfinance error fetching expiration dates for ticker '{ticker}': {yf_err}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching expiration dates for ticker '{ticker}': {e}")
        return []

def fetch_option_chain(ticker: str, expiration: str) -> pd.DataFrame:
    """
    Fetches the option chain for a given ticker and expiration date.

    :param ticker: Stock ticker symbol.
    :param expiration: Expiration date of the options.
    :return: DataFrame containing combined calls and puts data.
    """
    try:
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts

        if calls.empty and puts.empty:
            logger.warning(f"No options data found for ticker '{ticker}' on expiration date '{expiration}'.")
            return pd.DataFrame()

        # Add metadata columns
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        calls['expiration_date'] = expiration
        puts['expiration_date'] = expiration
        calls['fetched_at'] = timestamp
        puts['fetched_at'] = timestamp

        combined = pd.concat([calls, puts], ignore_index=True)
        return combined

    except Exception as e:
        logger.error(f"Error fetching options for ticker '{ticker}' on '{expiration}': {e}")
        return pd.DataFrame()

def fetch_and_save_options(ticker: str, start_date: str, end_date: str, save_dir: str) -> None:
    """
    Fetches and saves options data for a single ticker using parallel processing.

    :param ticker: Stock ticker symbol.
    :param start_date: Start date for fetching options.
    :param end_date: End date for fetching options.
    :param save_dir: Directory to save the options data.
    """
    try:
        stock = yf.Ticker(ticker)
        expiration_dates = stock.options
        if not expiration_dates:
            logger.error(f"No expiration dates found for ticker '{ticker}'.")
            return

        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Fetching options data for ticker '{ticker}'.")
        print(f"{Fore.GREEN}Starting to fetch options data for {ticker}...{Style.RESET_ALL}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_exp = {executor.submit(fetch_option_chain, ticker, exp): exp for exp in expiration_dates}
            for future in tqdm(as_completed(future_to_exp), total=len(expiration_dates), desc=f"Fetching options for {ticker}", unit="expiration"):
                exp = future_to_exp[future]
                data = future.result()
                if not data.empty:
                    file_path = os.path.join(save_dir, f"{ticker}_options_{exp}.csv")
                    data.to_csv(file_path, index=False, mode='a', chunksize=1000)
                    logger.info(f"Successfully saved options data to {file_path}")
        
        print(f"{Fore.GREEN}Completed fetching options data for {ticker}.{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Unexpected error in fetch_and_save_options for '{ticker}': {e}")
        print(f"{Fore.RED}Failed to fetch options data for {ticker}. Check logs for details.{Style.RESET_ALL}")

def fetch_and_save_options_chain(ticker: str) -> None:
    """
    Fetches the entire options chain for the given ticker and saves it as options.json
    in the corresponding data/{TICKER}/ directory.

    :param ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)
        expiration_dates = stock.options

        if not expiration_dates:
            logger.warning(f"No expiration dates found for ticker '{ticker}'.")
            return

        options_chain = {}

        for exp in expiration_dates:
            try:
                option_data = stock.option_chain(exp)
                options_chain[exp] = {
                    'calls': option_data.calls.to_dict(orient='records'),
                    'puts': option_data.puts.to_dict(orient='records')
                }
                logger.info(f"Fetched options data for '{ticker}' on expiration '{exp}'.")
            except Exception as e:
                logger.error(f"Error fetching options for ticker '{ticker}' on '{exp}': {e}")

        if not options_chain:
            logger.warning(f"No options data fetched for ticker '{ticker}'.")
            return

        save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{ticker}_options.json")

        with open(file_path, 'w') as json_file:
            json.dump(options_chain, json_file, indent=4)

        # Validate the saved JSON file
        with open(file_path, 'r') as json_file:
            try:
                json.load(json_file)
                logger.info(f"Successfully saved options chain to '{file_path}'.")
            except json.JSONDecodeError as e:
                logger.error(f"Validation failed for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Unexpected error in fetch_and_save_options_chain for '{ticker}': {e}")