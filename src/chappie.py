"""
chappie.py

Main script orchestrating data fetching, saving, and visualization for the Chappie Stock Analyzer.
Enhanced with progress bars for eacho data-fetching operation using tqdm, real-time logging of each step's status,
and consistentr logging format across modules.
"""

import os
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, Style
from tqdm import tqdm
import json

from logger import get_logger  # Centralized logger
from stock_fetcher import fetch_historical_data
from options_fetcher import fetch_and_save_options_chain
from fundamentals_fetcher import fetch_and_save_fundamentals
from news_fetcher import fetch_and_save_news
from sector_fetcher import fetch_and_save_sector_and_peers
from analyst_fetcher import fetch_and_save_analyst_data
from institutional_fetcher import fetch_and_save_institutional_data
from macro_fetcher import fetch_and_save_macro_data

# Import visualization functions from the new modules
from visualizer.visualizer import visualize_data

# Initialize logger
logger = get_logger('chappie', 'logs/chappie.log')


def create_ticker_directory(ticker: str) -> str:
    """
    Creates the base directory for a given ticker.
    """
    ticker_upper = ticker.upper()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_dir = os.path.join(project_root, 'data', ticker_upper)
    os.makedirs(base_dir, exist_ok=True)
    logger.debug(f"Directory ensured: {base_dir}")
    return base_dir


def main():
    """
    Main function to run the Chappie Stock Analyzer.
    """
    # Prompt the user for tickers
    user_input = input("Enter stock ticker symbols separated by commas: ")
    tickers = [ticker.strip().upper() for ticker in user_input.split(",")]

    # Initialize directories for each ticker
    directories = {ticker: create_ticker_directory(ticker) for ticker in tickers}

    # Initialize data storage
    stock_data_dict = {}

    # 1. Fetch historical stock data with progress bar
    logger.info("Starting to fetch historical stock data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(fetch_historical_data, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker), desc="Fetching Stock Data"):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                stock_data_dict[ticker] = data
                logger.info(f"Successfully fetched historical data for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch historical data for '{ticker}': {e}")

    # 2. Save stock data
    logger.info("Starting to save historical stock data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for ticker, data in stock_data_dict.items():
            save_dir = directories[ticker]
            futures.append(
                executor.submit(
                    data.to_csv,
                    os.path.join(save_dir, f"{ticker}_stock_data.csv"),
                    index=False,
                    encoding='utf-8'
                )
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving Stock Data"):
            try:
                future.result()
                logger.info("Stock data saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save stock data: {e}")

    # 3. Fetch options data with progress bar
    logger.info("Starting to fetch options data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_and_save_options_chain, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Options Data"):
            ticker = futures[future]
            try:
                future.result()
                logger.info(f"Successfully fetched options data for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch options data for '{ticker}': {e}")

    # 4. Fetch fundamentals data with progress bar
    logger.info("Starting to fetch fundamentals data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_and_save_fundamentals, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Fundamentals"):
            ticker = futures[future]
            try:
                future.result()
                logger.info(f"Successfully fetched fundamentals data for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals data for '{ticker}': {e}")

    # 5. Fetch news data with progress bar
    logger.info("Starting to fetch news data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_and_save_news, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching News"):
            ticker = futures[future]
            try:
                future.result()
                logger.info(f"Successfully fetched news data for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch news data for '{ticker}': {e}")

    # 6. Fetch sector and peers data with progress bar
    logger.info("Starting to fetch sector and peers data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_and_save_sector_and_peers, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Sector and Peers"):
            ticker = futures[future]
            try:
                future.result()
                logger.info(f"Successfully fetched sector and peers data for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch sector and peers data for '{ticker}': {e}")

    # 7. Fetch macroeconomic data
    for ticker in tickers:
        base_dir = directories[ticker]
        sector_file = os.path.join(base_dir, f"{ticker}_sector.json")
        if os.path.exists(sector_file):
            with open(sector_file, 'r') as f:
                sector_data = json.load(f)
            sector = sector_data.get('sector')
        else:
            sector = None
            logger.warning(f"Sector data not found for '{ticker}'. Cannot fetch macroeconomic indicators.")
        if sector:
            logger.info(f"Fetching macroeconomic data for sector '{sector}'.")
            fetch_and_save_macro_data(ticker, sector)
        else:
            logger.warning(f"Skipping macroeconomic data fetch for '{ticker}' due to missing sector information.")

    # 8. Fetch analyst ratings data
    logger.info("Starting to fetch analyst ratings data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_and_save_analyst_data, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Analyst Ratings"):
            ticker = futures[future]
            try:
                future.result()
                logger.info(f"Successfully fetched analyst ratings for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch analyst ratings for '{ticker}': {e}")

    # 9. Fetch institutional holdings data
    logger.info("Starting to fetch institutional holdings data.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_and_save_institutional_data, ticker): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Institutional Holdings"):
            ticker = futures[future]
            try:
                future.result()
                logger.info(f"Successfully fetched institutional holdings for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to fetch institutional holdings for '{ticker}': {e}")

    logger.info("Data fetching and saving completed.")

    # 10. Visualize data with progress bar
    logger.info("Starting data visualization.")
    for ticker in tqdm(tickers, desc="Visualizing Data"):
        data = stock_data_dict.get(ticker)
        if data is not None and not data.empty:
            base_dir = directories[ticker]
            sector_file = os.path.join(base_dir, f"{ticker}_sector.json")
            peers_file = os.path.join(base_dir, f"{ticker}_peers.json")
            sector_data = None
            peers_data = None
            if os.path.exists(sector_file):
                with open(sector_file, 'r') as f:
                    sector_data = json.load(f)
            if os.path.exists(peers_file):
                with open(peers_file, 'r') as f:
                    peers_data = json.load(f)
            try:
                visualize_data(data, ticker, sector_data, peers_data)
                logger.info(f"Visualization completed for '{ticker}'.")
            except Exception as e:
                logger.error(f"Failed to visualize data for '{ticker}': {e}")
        else:
            logger.warning(f"No data available to visualize for '{ticker}'.")

    logger.info("Chappie Stock Analyzer completed successfully.")

if __name__ == "__main__":
    main()

def visualize_data(
    stock_data: pd.DataFrame,
    ticker: str,
    sector_data: Dict[str, Any] = None,
    peers_data: List[Dict[str, Any]] = None,
    analyst_data: List[Dict[str, Any]] = None,
    holdings_data: Dict[str, Any] = None,
    macro_data: Dict[str, Any] = None,
    news_data: List[Dict[str, Any]] = None
):
    try:
        # Initialize progress manager for visualization tasks
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
            console=console
        )

        with progress:
            # Add tasks to the progress manager
            task_ids = {
                "sector": progress.add_task("Visualizing Sector Data", total=1),
                "fundamentals": progress.add_task("Visualizing Fundamentals", total=1),
                "news": progress.add_task("Visualizing News Articles", total=1),
                "stock": progress.add_task("Visualizing Stock Data", total=1),
                "analyst": progress.add_task("Visualizing Analyst Ratings", total=1),
                "holdings": progress.add_task("Visualizing Institutional Holdings", total=1)
            }

            # Display sector and industry data
            sector_file = os.path.join(data_dir, f"{ticker}_sector.json")
            if os.path.exists(sector_file):
                with open(sector_file, 'r') as f:
                    sector_data = json.load(f)
                display_sector_and_industry(sector_data, ticker)
                progress.update(task_ids["sector"], advance=1)
            else:
                logger.warning(f"Sector data not found for '{ticker}'.")
                progress.update(task_ids["sector"], advance=1, description="Sector Data Not Found")

            # Display fundamentals summary
            fundamentals_file = os.path.join(data_dir, f"{ticker}_fundamentals.json")
            if os.path.exists(fundamentals_file):
                with open(fundamentals_file, 'r') as f:
                    fundamentals = json.load(f)
                display_fundamentals(fundamentals, ticker)
                progress.update(task_ids["fundamentals"], advance=1)
            else:
                logger.warning(f"Fundamentals data not found for '{ticker}'.")
                progress.update(task_ids["fundamentals"], advance=1, description="Fundamentals Data Not Found")

            # Display news summary
            news_file = os.path.join(data_dir, f"{ticker}_news.json")
            top_news_headline = None
            if os.path.exists(news_file):
                with open(news_file, 'r') as f:
                    news_data = json.load(f)
                display_news_summary(news_data, ticker)
                if news_data:
                    top_news_headline = news_data[0]['headline']
                progress.update(task_ids["news"], advance=1)
            else:
                logger.warning(f"News data not found for '{ticker}'.")
                progress.update(task_ids["news"], advance=1, description="News Data Not Found")

            # Display stock price metrics
            latest_price = stock_data['Close'].iloc[-1] if not stock_data.empty else None
            six_month_high = stock_data['Close'].max() if not stock_data.empty else None
            six_month_low = stock_data['Close'].min() if not stock_data.empty else None
            display_stock_price_metrics(stock_data, ticker)
            progress.update(task_ids["stock"], advance=1)

            # Display analyst ratings
            analyst_file = os.path.join(data_dir, f"{ticker}_analyst_ratings.json")
            if os.path.exists(analyst_file):
                with open(analyst_file, 'r') as f:
                    analyst_data = json.load(f)
                display_analyst_ratings(analyst_data, ticker)
                progress.update(task_ids["analyst"], advance=1)
            else:
                logger.warning(f"Analyst ratings data not found for '{ticker}'.")
                progress.update(task_ids["analyst"], advance=1, description="Analyst Ratings Not Found")

            # Display institutional holdings
            holdings_file = os.path.join(data_dir, f"{ticker}_institutional_holdings.json")
            if os.path.exists(holdings_file):
                with open(holdings_file, 'r') as f:
                    holdings_data = json.load(f)
                display_institutional_holdings(holdings_data, ticker)
                progress.update(task_ids["holdings"], advance=1)
            else:
                logger.warning(f"Institutional holdings data not found for '{ticker}'.")
                progress.update(task_ids["holdings"], advance=1, description="Institutional Holdings Not Found")

            # Display final summary
            display_final_summary(
                ticker,
                latest_price=latest_price,
                six_month_high=six_month_high,
                six_month_low=six_month_low,
                top_news_headline=top_news_headline
            )

        logger.info(f"All visualizations completed for ticker '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to visualize data for ticker '{ticker}': {e}")
        console.print(f"[red]Failed to visualize data for ticker '{ticker}'. Check logs for details.[/red]")