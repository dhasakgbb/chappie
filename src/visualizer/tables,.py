# visualizer/tables.py

import logging
from rich.console import Console
from rich.table import Table
from rich import box

# Initialize rich's Console
console = Console()

# Initialize logger for this module using the centralized logger
logger = logging.getLogger('visualizer')

# Import utility functions
from visualizer.utils import display_rich_table

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