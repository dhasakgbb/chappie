# src/visualizer/progress.py

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

class ProgressManager:
    """
    Manages real-time progress updates for various tasks using Rich's Progress.
    """
    def __init__(self, task_descriptions: list):
        """
        Initializes the ProgressManager with the given task descriptions.

        :param task_descriptions: List of task descriptions to display progress for.
        """
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
            console=console
        )
        self.task_ids = {desc: self.progress.add_task(desc, total=100) for desc in task_descriptions}

    def start(self):
        """
        Starts the progress display.
        """
        self.progress.start()

    def update(self, task_description: str, advance_by: int = 1):
        """
        Updates the progress for a specific task.

        :param task_description: Description of the task to update.
        :param advance_by: Amount to advance the progress by.
        """
        if task_description in self.task_ids:
            self.progress.update(self.task_ids[task_description], advance=advance_by)

    def stop(self):
        """
        Stops the progress display.
        """
        self.progress.stop()