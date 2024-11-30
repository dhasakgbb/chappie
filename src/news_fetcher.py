# src/news_fetcher.py

"""
news_fetcher.py

This module provides functionality to fetch and save the latest news articles for a given stock ticker.
It uses yfinance to retrieve news articles and saves them as a JSON file in the assigned ticker directory.
A helper function filters out duplicate or outdated articles. Logging is used to trace the progress
of fetching and saving news data.
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf

from logger import get_logger  # Import the centralized logger

# Initialize logger for this module
logger = get_logger('news_fetcher', 'news_fetcher.log')


def filter_news_articles(
    new_articles: List[Dict[str, Any]],
    existing_articles: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Filters out duplicate or outdated news articles.

    :param new_articles: List of newly fetched articles.
    :param existing_articles: List of articles already saved.
    :return: Filtered list of articles to be saved.
    """
    existing_urls = {article['url'] for article in existing_articles}
    one_week_ago = datetime.now() - timedelta(days=7)

    filtered_articles = []
    for article in new_articles:
        article_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
        article_url = article.get('link')
        if article_url not in existing_urls and article_date >= one_week_ago:
            filtered_articles.append(article)
        else:
            logger.debug(f"Skipping duplicate or outdated article: '{article.get('title', 'No Title')}'")

    return filtered_articles


def fetch_and_save_news(ticker: str) -> None:
    """
    Fetches the latest news articles for the given ticker using yfinance and saves them as news.json
    in the corresponding data/{TICKER}/ directory.

    :param ticker: Stock ticker symbol.
    """
    try:
        logger.info(f"Fetching news articles for ticker '{ticker}'.")
        stock = yf.Ticker(ticker)
        news_data = stock.news

        if not news_data:
            logger.warning(f"No news data found for ticker '{ticker}'.")
            return

        save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{ticker}_news.json")

        if os.path.isfile(file_path):
            with open(file_path, 'r') as json_file:
                existing_articles = json.load(json_file)
            logger.info(f"Loaded existing news articles for ticker '{ticker}'.")
        else:
            existing_articles = []

        filtered_articles = filter_news_articles(news_data, existing_articles)

        if not filtered_articles:
            logger.info(f"No new articles to add for ticker '{ticker}'.")
            return

        formatted_articles = []
        for article in filtered_articles:
            try:
                publication_time = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                formatted_article = {
                    'headline': article.get('title', ''),
                    'source': article.get('publisher', ''),
                    'publication_date': publication_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'url': article.get('link', '')
                }
                formatted_articles.append(formatted_article)
            except Exception as e:
                logger.error(f"Error processing article '{article.get('title', 'No Title')}': {e}")

        combined_articles = existing_articles + formatted_articles

        # Save the combined articles to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(combined_articles, json_file, indent=4)

        # Validate the saved JSON file
        with open(file_path, 'r') as json_file:
            try:
                json.load(json_file)
                logger.info(f"Successfully saved {len(formatted_articles)} new articles for ticker '{ticker}'.")
            except json.JSONDecodeError as e:
                logger.error(f"Validation failed for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Error fetching news for ticker '{ticker}': {e}")