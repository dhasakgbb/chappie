# visualizer/utils.py

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.align import Align
import logging

# Initialize rich's Console
console = Console()

# Initialize logger for this module using the centralized logger
logger = logging.getLogger('visualizer')

def format_number(value: Any) -> str:
    """
    Formats numerical values for readability.
    Formats percentages, large numbers, and floats to two decimal places.
    """
    if isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float):
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
    if any(keyword in header.lower() for keyword in keywords):
        stripped_value = formatted_value.replace('%', '').replace(',', '').replace('$', '')
        try:
            numeric_value = float(stripped_value)
            if numeric_value > 0:
                return f"[green]{formatted_value}[/green]"
            elif numeric_value < 0:
                return f"[red]{formatted_value}[/red]"
            else:
                return formatted_value
        except ValueError:
            return formatted_value
    else:
        return formatted_value

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
        headers = list(data[0].keys())
        table = Table(title=title, box=box.DOUBLE_EDGE, header_style="bold magenta")

        for header in headers:
            table.add_column(header, overflow="fold", justify="right")

        for row in data:
            formatted_row = []
            for header in headers:
                value = row.get(header, "")
                formatted_value = format_number(value) if isinstance(value, (int, float)) else str(value)
                formatted_value = highlight_value(formatted_value, header)
                formatted_row.append(formatted_value)
            table.add_row(*formatted_row)

        if description:
            console.print(Align.center(Text(description, style="italic cyan")))

        console.print(table)

    except Exception as e:
        logger.error(f"Failed to display rich table '{title}': {e}")
        console.print(f"[red]Failed to display table '{title}'. Check logs for details.[/red]")