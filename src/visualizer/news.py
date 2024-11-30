# visualizer/news.py

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