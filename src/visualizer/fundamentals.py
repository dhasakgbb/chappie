# visualizer/fundamentals.py

import logging
from rich.console import Console
from rich.table import Table
from rich import box

# Initialize rich's Console
console = Console()

# Initialize logger for this module using the centralized logger
logger = logging.getLogger('visualizer')

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