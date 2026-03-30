"""Core logging and progress bar implementation.

Implements ``get_logger`` for library-wide logging configuration and
``ManualProgressBar`` for manual progress tracking during long operations.

Classes:
    ManualProgressBar: Manual progress bar with verbosity control.

Functions:
    get_logger: Creates and configures a logger instance.

Examples:
    >>> from auf.log.log import get_logger, ManualProgressBar
    >>> logger = get_logger(verbosity=2)
    >>> logger.debug("Debug message enabled")
"""

import logging
import typing as tp

from tqdm import tqdm

LEVELS_MAP = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

FORMATTER_MAP = {
    0: logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s",
        "%H:%M:%S",
    ),
    1: logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s",
        "%H:%M:%S",
    ),
    2: logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(funcName)s - %(levelname)s - %(message)s",
        "%d-%m-%Y %H:%M:%S",
    ),
}


def get_logger(verbosity: int = 1) -> logging.Logger:
    """Create and return a configured logger for the AUF library.

    Configures a logger with console output and appropriate formatting
    based on verbosity level.

    Args:
        verbosity: Logging level (0=WARNING, 1=INFO, 2=DEBUG).
            Defaults to 1.

    Returns:
        Configured ``logging.Logger`` instance named 'auf'.

    Examples:
        >>> from auf.log import get_logger
        >>> logger = get_logger(verbosity=1)
        >>> logger.info("Library initialized")

    Notes:
        Replaces existing handlers on each call to ensure clean configuration.
        Falls back to INFO level if verbosity is not in {0, 1, 2}.
    """
    logger = logging.getLogger("auf")

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    level = LEVELS_MAP.get(verbosity, logging.INFO)
    formatter = FORMATTER_MAP.get(verbosity)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class ManualProgressBar:
    """Manual progress bar wrapper around ``tqdm``.

    Provides a progress bar that can be enabled or disabled via verbosity
    parameter. Supports manual updates and description changes.

    Attributes:
        _verbosity (int): Controls bar visibility (0=disabled, >0=enabled).
        _pbar (Optional[tqdm]): Underlying ``tqdm`` instance or None.

    Examples:
        >>> from auf.log import ManualProgressBar

        >>> with ManualProgressBar(total=10, description="Fitting") as bar:
        ...     for i in range(10):
        ...         # ... some processing ...
        ...         bar.update(1)
        ...         bar.update_description(f"Step {i+1}")

    Notes:
        Implements context manager protocol for automatic closing.
        All methods are safe no-ops when verbosity=0.
    """

    def __init__(
        self,
        total: int = 100,
        description: str = "Progress",
        verbosity: int = 1,
    ):
        """Initialize ManualProgressBar.

        Args:
            total: Total number of expected iterations. Defaults to 100.
            description: Prefix text displayed before the progress bar.
                Defaults to "Progress".
            verbosity: If 0, progress bar is disabled. Defaults to 1.

        Examples:
            >>> from auf.log import ManualProgressBar
            >>> bar = ManualProgressBar(total=50, description="Training", verbosity=1)
            >>> bar.close()
        """
        self._verbosity = verbosity
        self._pbar: tp.Optional[tqdm] = None

        if self._verbosity > 0:
            self._pbar = tqdm(
                total=total,
                desc=description,
                bar_format="{desc}: {percentage:.0f}%|{bar}|",
                disable=(verbosity == 0),
            )

    def update(self, value: int):
        """Advance the progress bar by specified amount.

        Args:
            value: Number of steps to advance.

        Examples:
            >>> from auf.log import ManualProgressBar
            >>> bar = ManualProgressBar(total=10, verbosity=1)
            >>> bar.update(5)
            >>> bar.close()
        """
        if self._pbar is not None:
            self._pbar.update(value)

    def update_description(self, new_description: str):
        """Update the description text of the progress bar.

        Args:
            new_description: New text to display before the progress bar.

        Examples:
            >>> from auf.log import ManualProgressBar
            >>> bar = ManualProgressBar(total=10, verbosity=1)
            >>> bar.update_description("Processing step 1")
            >>> bar.close()
        """
        if self._pbar is not None:
            self._pbar.set_description(new_description, refresh=True)

    def close(self):
        """Close the progress bar and clean up resources.

        Examples:
            >>> from auf.log import ManualProgressBar
            >>> bar = ManualProgressBar(total=10, verbosity=1)
            >>> bar.update(10)
            >>> bar.close()
        """
        if self._pbar is not None:
            self._pbar.close()
