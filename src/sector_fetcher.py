# src/sector_fetcher.py

"""
sector_fetcher.py

This module provides functionality to fetch and save the sector, industry, and peers data for a given stock ticker.
It uses yfinance to retrieve the necessary information and saves it as JSON files in the assigned ticker directory.
"""

import os
import json
from typing import Dict, Any, List
import yfinance as yf

from logger import get_logger  # Import the centralized logger

# Initialize logger for this module
logger = get_logger('sector_fetcher', 'sector_fetcher.log')


def fetch_and_save_sector_and_peers(ticker: str) -> None:
    """
    Fetches the sector, industry, and peers data for the given ticker using yfinance and saves it as sector.json and peers.json
    in the corresponding data/{TICKER}/ directory.

    :param ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)

        # Retrieve fundamental data from the 'info' attribute
        info = stock.info

        if not info:
            logger.warning(f"No fundamental data found for ticker '{ticker}'.")
            return

        # Extract sector and industry information
        sector = info.get('sector')
        industry = info.get('industry')

        if not sector or not industry:
            logger.warning(f"Sector or industry information missing for ticker '{ticker}'.")
        else:
            logger.info(f"Sector: {sector}, Industry: {industry} for '{ticker}'.")

        # Get the absolute path to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Define the save directory
        save_dir = os.path.join(project_root, 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)

        # Save sector and industry information to sector.json
        sector_data = {
            'ticker': ticker.upper(),
            'sector': sector,
            'industry': industry
        }

        sector_file_path = os.path.join(save_dir, f"{ticker}_sector.json")

        # Fetch peers data
        # yfinance does not provide peers directly, so we need an alternative approach

        # Alternative approach: Use the industry to find other tickers in the same industry
        # This requires access to a database or API that maps industries to tickers
        # For the purposes of this example, we'll use a placeholder function

        peers_list = get_peers_from_industry(industry, exclude_ticker=ticker.upper())

        # Fetch metrics for each peer
        peers_data = []
        for peer_ticker in peers_list:
            try:
                peer_stock = yf.Ticker(peer_ticker)
                peer_info = peer_stock.info

                if not peer_info:
                    logger.warning(f"No data found for peer '{peer_ticker}'.")
                    continue

                peer_data = {
                    'ticker': peer_ticker,
                    'companyName': peer_info.get('longName'),
                    'marketCap': peer_info.get('marketCap'),
                    'trailingPE': peer_info.get('trailingPE'),
                    'forwardPE': peer_info.get('forwardPE'),
                    'priceToBook': peer_info.get('priceToBook'),
                    'sector': peer_info.get('sector'),
                    'industry': peer_info.get('industry')
                }
                peers_data.append(peer_data)
            except Exception as e:
                logger.error(f"Error fetching data for peer '{peer_ticker}': {e}")

        # Save peers data to peers.json
        peers_file_path = os.path.join(save_dir, f"{ticker}_peers.json")

        # Save the data
        with open(sector_file_path, 'w') as json_file:
            json.dump(sector_data, json_file, indent=4)
        logger.info(f"Sector data saved to '{sector_file_path}'.")

        with open(peers_file_path, 'w') as json_file:
            json.dump(peers_data, json_file, indent=4)
        logger.info(f"Peers data saved to '{peers_file_path}'.")

    except Exception as e:
        logger.error(f"Unexpected error in fetch_and_save_sector_and_peers for '{ticker}': {e}")


def get_peers_from_industry(industry: str, exclude_ticker: str) -> List[str]:
    """
    Retrieves a list of peer tickers in the same industry.
    In a real-world scenario, this function would query a database or an API to get the data.

    :param industry: Industry name.
    :param exclude_ticker: Ticker symbol to exclude from the peers list.
    :return: List of peer ticker symbols.
    """
    # Placeholder implementation
    # In an actual implementation, retrieve a list of tickers in the same industry
    # For example, use an external API or a financial database

    # Example industry peers (static data for demonstration)
    industry_peers = {
        'Information Technology': ['MSFT', 'GOOGL', 'AMZN', 'IBM', 'ORCL'],
        'Health Care': ['JNJ', 'PFE', 'MRK', 'ABBV', 'TMO'],
        'Financials': ['JPM', 'BAC', 'C', 'WFC', 'GS']
        # Add more industries as needed
    }

    peers_list = industry_peers.get(industry, [])
    if exclude_ticker in peers_list:
        peers_list.remove(exclude_ticker)
    return peers_list