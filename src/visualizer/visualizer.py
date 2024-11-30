"""
visualizer.py

Provides functions to visualize stock data and display rich tables using the rich library.
Includes functions to dynamically display data in tables with formatted numerical values and highlighted significant values.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, ProgressColumn
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
import os
import json

from logger import get_logger  # Import the centralized logger
from visualizer.utils import display_rich_table  # Import from utils.py
from visualizer.tables import display_summary, display_metrics_table, display_sector_and_industry, display_fundamentals, display_news_summary, display_stock_price_metrics, display_analyst_ratings, display_institutional_holdings, display_final_summary
from visualizer.alerts import display_alerts
from visualizer.news import summarize_news, visualize_news
from visualizer.progress import ProgressManager

# Initialize rich's Console
console = Console()

# Initialize logger for this module using the centralized logger
logger = get_logger('visualizer', 'logs/visualizer.log')


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
        progress_manager = ProgressManager([
            "Visualizing Sector Data",
            "Visualizing Fundamentals",
            "Visualizing News Articles",
            "Visualizing Stock Data",
            "Visualizing Analyst Ratings",
            "Visualizing Institutional Holdings"
        ])
        progress_manager.start()

        # Display sector and industry data
        sector_file = os.path.join(data_dir, f"{ticker}_sector.json")
        if os.path.exists(sector_file):
            with open(sector_file, 'r') as f:
                sector_data = json.load(f)
            display_sector_and_industry(sector_data, ticker)
            progress_manager.update("Visualizing Sector Data")
        else:
            logger.warning(f"Sector data not found for '{ticker}'.")
            progress_manager.update("Visualizing Sector Data")

        # Display fundamentals summary
        fundamentals_file = os.path.join(data_dir, f"{ticker}_fundamentals.json")
        if os.path.exists(fundamentals_file):
            with open(fundamentals_file, 'r') as f:
                fundamentals = json.load(f)
            display_fundamentals(fundamentals, ticker)
            progress_manager.update("Visualizing Fundamentals")
        else:
            logger.warning(f"Fundamentals data not found for '{ticker}'.")
            progress_manager.update("Visualizing Fundamentals")

        # Display news summary
        news_file = os.path.join(data_dir, f"{ticker}_news.json")
        top_news_headline = None
        if os.path.exists(news_file):
            with open(news_file, 'r') as f:
                news_data = json.load(f)
            display_news_summary(news_data, ticker)
            if news_data:
                top_news_headline = news_data[0]['headline']
            progress_manager.update("Visualizing News Articles")
        else:
            logger.warning(f"News data not found for '{ticker}'.")
            progress_manager.update("Visualizing News Articles")

        # Display stock price metrics
        latest_price = stock_data['Close'].iloc[-1] if not stock_data.empty else None
        six_month_high = stock_data['Close'].max() if not stock_data.empty else None
        six_month_low = stock_data['Close'].min() if not stock_data.empty else None
        display_stock_price_metrics(stock_data, ticker)
        progress_manager.update("Visualizing Stock Data")

        # Display analyst ratings
        analyst_file = os.path.join(data_dir, f"{ticker}_analyst_ratings.json")
        if os.path.exists(analyst_file):
            with open(analyst_file, 'r') as f:
                analyst_data = json.load(f)
            display_analyst_ratings(analyst_data, ticker)
            progress_manager.update("Visualizing Analyst Ratings")
        else:
            logger.warning(f"Analyst ratings data not found for '{ticker}'.")
            progress_manager.update("Visualizing Analyst Ratings")

        # Display institutional holdings
        holdings_file = os.path.join(data_dir, f"{ticker}_institutional_holdings.json")
        if os.path.exists(holdings_file):
            with open(holdings_file, 'r') as f:
                holdings_data = json.load(f)
            display_institutional_holdings(holdings_data, ticker)
            progress_manager.update("Visualizing Institutional Holdings")
        else:
            logger.warning(f"Institutional holdings data not found for '{ticker}'.")
            progress_manager.update("Visualizing Institutional Holdings")

        # Display final summary
        display_final_summary(
            ticker,
            latest_price=latest_price,
            six_month_high=six_month_high,
            six_month_low=six_month_low,
            top_news_headline=top_news_headline
        )

        progress_manager.stop()
        logger.info(f"All visualizations completed for ticker '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to visualize data for ticker '{ticker}': {e}")
        console.print(f"[red]Failed to visualize data for ticker '{ticker}'. Check logs for details.[/red]")


def display_summary(stock_data: pd.DataFrame, options_data_counts: Dict[str, int]) -> None:
    """
    Displays a summary of the stock data and options data.

    :param stock_data: DataFrame containing stock data.
    :param options_data_counts: Dictionary containing counts of calls and puts.
    """
    try:
        logger.info("Displaying summary of fetched data.")
        latest_price = stock_data['Close'].iloc[-1]
        six_month_high = stock_data['Close'].max()
        six_month_low = stock_data['Close'].min()
        total_options = options_data_counts.get('calls', 0) + options_data_counts.get('puts', 0)

        price_change = latest_price - stock_data['Close'].iloc[0]
        price_color = "green" if price_change >= 0 else "red"

        summary_table = Table(title="📈 Stock Summary", box=box.MINIMAL_DOUBLE_HEAD)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="")

        summary_table.add_row(
            "Latest Close Price",
            f"[{price_color}]{latest_price:.2f}[/{price_color}]"
        )
        summary_table.add_row(
            "6-Month High",
            f"[bold green]${six_month_high:,.2f}[/bold green]"
        )
        summary_table.add_row(
            "6-Month Low",
            f"[bold red]${six_month_low:,.2f}[/bold red]"
        )
        summary_table.add_row(
            "Total Options Contracts",
            f"{total_options}"
        )

        console.print(summary_table)
        logger.info("Summary displayed successfully.")

    except Exception as e:
        logger.error(f"Failed to display summary: {e}")
        console.print(f"[red]Failed to display summary. Check logs for details.[/red]")


def display_metrics_table(stock_data: pd.DataFrame, options_data_counts: Dict[str, int]) -> None:
    """
    Displays a table summarizing key stock metrics and options data counts.

    :param stock_data: DataFrame containing stock data.
    :param options_data_counts: Dictionary containing counts of calls and puts.
    """
    try:
        logger.info("Displaying metrics table.")
        metrics_table = Table(title="🔍 Key Metrics", box=box.MINIMAL_DOUBLE_HEAD)
        metrics_table.add_column("Metric", style="bold magenta")
        metrics_table.add_column("Value", style="")

        # Add stock metrics
        latest_volume = stock_data['Volume'].iloc[-1]
        open_price = stock_data['Open'].iloc[-1]
        high_price = stock_data['High'].iloc[-1]
        low_price = stock_data['Low'].iloc[-1]

        metrics_table.add_row("Latest Volume", f"{latest_volume:,}")
        metrics_table.add_row("Open Price", f"${open_price:.2f}")
        metrics_table.add_row("High Price", f"${high_price:.2f}")
        metrics_table.add_row("Low Price", f"${low_price:.2f}")

        # Add options metrics
        calls = options_data_counts.get('calls', 0)
        puts = options_data_counts.get('puts', 0)

        metrics_table.add_row("Total Calls", f"{calls}")
        metrics_table.add_row("Total Puts", f"{puts}")

        console.print(metrics_table)
        logger.info("Metrics table displayed successfully.")

    except Exception as e:
        logger.error(f"Failed to display metrics table: {e}")
        console.print(f"[red]Failed to display metrics table. Check logs for details.[/red]")


def display_alerts(stock_data: pd.DataFrame) -> None:
    """
    Displays alerts for significant stock performance changes.

    :param stock_data: DataFrame containing stock data.
    """
    try:
        logger.info("Checking for significant performance changes.")
        latest_close = stock_data['Close'].iloc[-1]
        previous_close = stock_data['Close'].iloc[-2]
        change_percent = ((latest_close - previous_close) / previous_close) * 100

        if change_percent >= 5:
            alert = Panel.fit(
                f"🚀 [bold green]Significant Increase[/bold green]: {change_percent:.2f}% since yesterday.",
                style="bold green"
            )
            console.print(alert)
            logger.info("Displayed significant increase alert.")

        elif change_percent <= -5:
            alert = Panel.fit(
                f"📉 [bold red]Significant Decrease[/bold red]: {change_percent:.2f}% since yesterday.",
                style="bold red"
            )
            console.print(alert)
            logger.info("Displayed significant decrease alert.")

    except Exception as e:
        logger.error(f"Failed to display alerts: {e}")
        console.print(f"[red]Failed to display alerts. Check logs for details.[/red]")


def display_rich_table(data: List[Dict[str, Any]], title: str, description: str = None) -> None:
    """
    Displays a list of dictionaries as a table using rich.Table.

    :param data: List of dictionaries containing the data to display.
    :param title: Title of the table.
    :param description: Optional description to display above the table.
    """
    if not data:
        console.print(f"[red]No data available to display in the table '{title}'.[/red]")
        return

    try:
        # Dynamically get the headers from the keys of the first dictionary
        headers = list(data[0].keys())

        # Create a Rich Table with the provided title
        table = Table(title=title, box=box.DOUBLE_EDGE, header_style="bold magenta")

        # Add columns to the table based on headers
        for header in headers:
            table.add_column(header, overflow="fold", justify="right")

        # Add rows to the table
        for row in data:
            formatted_row = []
            for header in headers:
                value = row.get(header, "")
                # Format numerical values
                if isinstance(value, (int, float)):
                    formatted_value = format_number(value)
                else:
                    formatted_value = str(value)
                # Highlight significant values
                formatted_value = highlight_value(formatted_value, header)
                formatted_row.append(formatted_value)
            table.add_row(*formatted_row)

        # If description is provided, print it above the table
        if (description):
            console.print(Align.center(Text(description, style="italic cyan")))

        console.print(table)

    except Exception as e:
        logger.error(f"Failed to display rich table '{title}': {e}")
        console.print(f"[red]Failed to display table '{title}'. Check logs for details.[/red]")


def format_number(value: Any) -> str:
    """
    Formats numerical values for readability.
    Formats percentages, large numbers, and floats to two decimal places.
    """
    if isinstance(value, int):
        # Format large integers with commas
        return f"{value:,}"
    elif isinstance(value, float):
        # For percentages between -1 and 1, format as percentage
        if -1 < value < 1 and value != 0:
            return f"{value:.2%}"
        else:
            return f"{value:,.2f}"
    else:
        return str(value)


def highlight_value(formatted_value: str, header: str) -> str:
    """
    Highlights significant values based on the header keyword.
    Color codes price changes or volatility.

    :param formatted_value: The value after formatting.
    :param header: The column header.
    :return: The formatted value with color highlights if applicable.
    """
    keywords = ['change', 'volatility', 'delta', 'price change', 'pct change', '% change', 'growth', 'return']

    # Check if the header indicates a significant value
    if any(keyword in header.lower() for keyword in keywords):
        # Strip symbols to attempt conversion to float
        stripped_value = formatted_value.replace('%', '').replace(',', '').replace('$', '')
        try:
            numeric_value = float(stripped_value)
            if (numeric_value > 0):
                return f"[green]{formatted_value}[/green]"
            elif (numeric_value < 0):
                return f"[red]{formatted_value}[/red]"
            else:
                return formatted_value
        except ValueError:
            # If conversion fails, return the value as-is
            return formatted_value
    else:
        return formatted_value


def visualize_analyst_ratings(analyst_data, ticker):
    """
    Visualizes analyst ratings using a rich table.
    """
    if not analyst_data:
        logger.warning(f"No analyst data to visualize for '{ticker}'.")
        return

    display_rich_table(
        data=analyst_data,
        title=f"Analyst Ratings for {ticker}",
        description="Summary of analyst recommendations and price targets."
    )


def visualize_institutional_holdings(holdings_data, ticker):
    """
    Visualizes the top institutional holders using a rich table.
    """
    institutional_holders = holdings_data.get('institutional_holders', [])
    if not institutional_holders:
        logger.warning(f"No institutional holders data to visualize for '{ticker}'.")
        return

    display_rich_table(
        data=institutional_holders,
        title=f"Top Institutional Holders for {ticker}",
        description="Summary of top institutional holders and their holdings."
    )


def visualize_macro_data(macro_data, ticker):
    """
    Visualizes macroeconomic indicators using a rich table.
    """
    if not macro_data:
        logger.warning(f"No macroeconomic data to visualize for '{ticker}'.")
        return

    display_rich_table(
        data=[{'Indicator': k, 'Value': v} for k, v in macro_data.items()],
        title=f"Macroeconomic Indicators for {ticker}",
        description="Summary of relevant macroeconomic indicators."
    )


def visualize_news(news_data, ticker):
    """
    Visualizes news articles using a rich table.
    """
    if not news_data:
        logger.warning(f"No news data to visualize for '{ticker}'.")
        return

    display_rich_table(
        data=news_data,
        title=f"News for {ticker}",
        description="Summary of recent news articles."
    )


def summarize_news(news_data: List[Dict[str, Any]], ticker: str) -> None:
    """
    Summarizes news articles for the given ticker.
    :param news_data: List of dictionaries containing news articles.
    :param ticker: Stock ticker symbol.
    """
    try:
        if not news_data:
            logger.warning(f"No news data to summarize for '{ticker}'.")
            return

        num_articles = len(news_data)
        publication_dates = [article['publication_date'] for article in news_data]
        top_articles = news_data[:3]

        console.print(f"\n[bold cyan]News Summary for {ticker}[/bold cyan]")
        console.print(f"Total Articles: {num_articles}")
        console.print(f"Publication Dates: {', '.join(publication_dates)}")

        table = Table(title="Top 3 Articles", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Headline", style="bold magenta")
        table.add_column("Source", style="")
        table.add_column("URL", style="")

        for article in top_articles:
            table.add_row(article['headline'], article['source'], article['url'])

        console.print(table)
        logger.info(f"News summary displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to summarize news for '{ticker}': {e}")


def summarize_stock_data(stock_data: pd.DataFrame, ticker: str) -> None:
    """
    Summarizes stock price data for the given ticker.
    :param stock_data: DataFrame containing stock data.
    :param ticker: Stock ticker symbol.
    """
    try:
        if stock_data.empty:
            logger.warning(f"No stock data to summarize for '{ticker}'.")
            return

        latest_close = stock_data['Close'].iloc[-1]
        six_month_high = stock_data['Close'].max()
        six_month_low = stock_data['Close'].min()
        price_change_pct = ((latest_close - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100

        console.print(f"\n[bold cyan]Stock Price Summary for {ticker}[/bold cyan]")
        console.print(f"Latest Close Price: {latest_close:.2f}")
        console.print(f"6-Month High: {six_month_high:.2f}")
        console.print(f"6-Month Low: {six_month_low:.2f}")
        console.print(f"Price Change Percentage: {price_change_pct:.2f}%")

        logger.info(f"Stock price summary displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to summarize stock data for '{ticker}': {e}")


def summarize_fundamentals(fundamentals: Dict[str, Any], ticker: str) -> None:
    """
    Summarizes fundamental data for the given ticker.
    :param fundamentals: Dictionary containing fundamental data.
    :param ticker: Stock ticker symbol.
    """
    try:
        if not fundamentals:
            logger.warning(f"No fundamental data to summarize for '{ticker}'.")
            return

        console.print(f"\n[bold cyan]Fundamentals Summary for {ticker}[/bold cyan]")

        table = Table(title="Fundamentals", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Metric", style="bold magenta")
        table.add_column("Value", style="")

        for key, value in fundamentals.items():
            table.add_row(key, str(value))

        console.print(table)
        logger.info(f"Fundamentals summary displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to summarize fundamentals for '{ticker}': {e}")


def display_sector_and_industry(sector_data: Dict[str, Any], ticker: str) -> None:
    """
    Displays the sector and industry information for the stock.
    :param sector_data: Dictionary containing sector and industry data.
    :param ticker: Stock ticker symbol.
    """
    try:
        if not sector_data:
            logger.warning(f"No sector data to display for '{ticker}'.")
            return

        console.print(f"\n[bold cyan]Sector and Industry Data for {ticker}[/bold cyan]")
        table = Table(box=box.SIMPLE)
        table.add_column("Ticker", style="bold magenta")
        table.add_column("Sector", style="")
        table.add_column("Industry", style="")

        table.add_row(sector_data.get('ticker', ''), sector_data.get('sector', ''), sector_data.get('industry', ''))

        console.print(table)
        logger.info(f"Sector and industry data displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display sector and industry data for '{ticker}': {e}")


def display_fundamentals(fundamentals: Dict[str, Any], ticker: str) -> None:
    """
    Displays fundamental data for the given ticker.
    :param fundamentals: Dictionary containing fundamental data.
    :param ticker: Stock ticker symbol.
    """
    try:
        if not fundamentals:
            logger.warning(f"No fundamental data to display for '{ticker}'.")
            return

        console.print(f"\n[bold cyan]Fundamentals Summary for {ticker}[/bold cyan]")
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="bold magenta")
        table.add_column("Value", style="")

        for key, value in fundamentals.items():
            table.add_row(key, str(value))

        console.print(table)
        logger.info(f"Fundamentals summary displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display fundamentals for '{ticker}': {e}")


def display_news_summary(news_data: List[Dict[str, Any]], ticker: str) -> None:
    """
    Displays a summary of news articles for the given ticker.
    :param news_data: List of dictionaries containing news articles.
    :param ticker: Stock ticker symbol.
    """
    try:
        if not news_data:
            logger.warning(f"No news data to display for '{ticker}'.")
            return

        console.print(f"\n[bold cyan]News Summary for {ticker}[/bold cyan]")
        num_articles = len(news_data)
        publication_dates = [article['publication_date'] for article in news_data]
        top_articles = news_data[:3]

        console.print(f"Total Articles: {num_articles}")
        console.print(f"Publication Dates: {', '.join(publication_dates)}")

        table = Table(title="Top 3 Articles", box=box.SIMPLE)
        table.add_column("Headline", style="bold magenta")
        table.add_column("Source", style="")
        table.add_column("URL", style="")

        for article in top_articles:
            table.add_row(article['headline'], article['source'], article['url'])

        console.print(table)
        logger.info(f"News summary displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display news summary for '{ticker}': {e}")


def display_stock_price_metrics(stock_data: pd.DataFrame, ticker: str) -> None:
    """
    Displays stock price metrics for the given ticker.
    :param stock_data: DataFrame containing stock data.
    :param ticker: Stock ticker symbol.
    """
    try:
        if stock_data.empty:
            logger.warning(f"No stock data to display for '{ticker}'.")
            return

        latest_close = stock_data['Close'].iloc[-1]
        six_month_high = stock_data['Close'].max()
        six_month_low = stock_data['Close'].min()
        price_change_pct = ((latest_close - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100

        console.print(f"\n[bold cyan]Stock Price Metrics for {ticker}[/bold cyan]")
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="bold magenta")
        table.add_column("Value", style="")

        table.add_row("Latest Close Price", f"{latest_close:.2f}")
        table.add_row("6-Month High", f"{six_month_high:.2f}")
        table.add_row("6-Month Low", f"{six_month_low:.2f}")
        table.add_row("Price Change Percentage", f"{price_change_pct:.2f}%")

        console.print(table)
        logger.info(f"Stock price metrics displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display stock price metrics for '{ticker}': {e}")


def display_analyst_ratings(analyst_data: List[Dict[str, Any]], ticker: str) -> None:
    """
    Displays analyst ratings for the given ticker.
    :param analyst_data: List of dictionaries containing analyst ratings.
    :param ticker: Stock ticker symbol.
    """
    try:
        if not analyst_data:
            logger.warning(f"No analyst data to display for '{ticker}'.")
            return

        console.print(f"\n[bold cyan]Analyst Ratings for {ticker}[/bold cyan]")
        table = Table(box=box.SIMPLE)
        table.add_column("Period", style="bold magenta")
        table.add_column("Strong Buy", style="")
        table.add_column("Buy", style="")
        table.add_column("Hold", style="")
        table.add_column("Sell", style="")
        table.add_column("Strong Sell", style="")

        for rating in analyst_data:
            table.add_row(
                rating.get('period', ''),
                str(rating.get('strongBuy', '')),
                str(rating.get('buy', '')),
                str(rating.get('hold', '')),
                str(rating.get('sell', '')),
                str(rating.get('strongSell', ''))
            )

        console.print(table)
        logger.info(f"Analyst ratings displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display analyst ratings for '{ticker}': {e}")


def display_institutional_holdings(holdings_data: Dict[str, Any], ticker: str) -> None:
    """
    Displays institutional holdings for the given ticker.
    :param holdings_data: Dictionary containing institutional holdings data.
    :param ticker: Stock ticker symbol.
    """
    try:
        institutional_holders = holdings_data.get('institutional_holders', [])
        if not institutional_holders:
            logger.warning(f"No institutional holders data to display for '{ticker}'.")
            return

        console.print(f"\n[bold cyan]Institutional Holdings for {ticker}[/bold cyan]")
        table = Table(box=box.SIMPLE)
        table.add_column("Holder", style="bold magenta")
        table.add_column("Percentage Held", style="")
        table.add_column("Shares", style="")
        table.add_column("Value", style="")

        for holder in institutional_holders:
            table.add_row(
                holder.get('Holder', ''),
                f"{holder.get('pctHeld', 0) * 100:.2f}%",
                f"{holder.get('Shares', 0):,}",
                f"${holder.get('Value', 0):,}"
            )

        console.print(table)
        logger.info(f"Institutional holdings displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display institutional holdings for '{ticker}': {e}")


def display_final_summary(
    ticker: str,
    latest_price: float = None,
    six_month_high: float = None,
    six_month_low: float = None,
    top_news_headline: str = None
) -> None:
    """
    Displays a final summary of the most important information for quick reference.
    :param ticker: Stock ticker symbol.
    :param latest_price: Latest closing price of the stock.
    :param six_month_high: 6-month high price of the stock.
    :param six_month_low: 6-month low price of the stock.
    :param top_news_headline: Headline of the top news article.
    """
    try:
        console.print(f"\n[bold cyan]Final Summary for {ticker}[/bold cyan]")

        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="bold magenta")
        table.add_column("Value", style="")

        table.add_row("Latest Close Price", f"{latest_price:.2f}" if latest_price is not None else "N/A")
        table.add_row("6-Month High", f"{six_month_high:.2f}" if six_month_high is not None else "N/A")
        table.add_row("6-Month Low", f"{six_month_low:.2f}" if six_month_low is not None else "N/A")
        table.add_row("Top News Headline", top_news_headline if top_news_headline else "N/A")

        console.print(table)
        logger.info(f"Final summary displayed for '{ticker}'.")

    except Exception as e:
        logger.error(f"Failed to display final summary for '{ticker}': {e}")