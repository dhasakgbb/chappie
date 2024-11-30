# visualizer/alerts.py

import logging
from rich.console import Console
from rich.panel import Panel

# Initialize rich's Console
console = Console()

# Initialize logger for this module using the centralized logger
logger = logging.getLogger('visualizer')

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