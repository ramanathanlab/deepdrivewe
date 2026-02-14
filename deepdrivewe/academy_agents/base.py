"""Base agent class for deepdrivewe Academy agents."""

from __future__ import annotations

import logging
from typing import Any

from academy.agent import Agent


class AcademyAgent(Agent):
    """Base class for all deepdrivewe Academy agents.

    This class extends Academy's Agent class and provides common
    functionality for all deepdrivewe agents, including:
    - Standardized logging
    - Error handling patterns
    - State management utilities

    All deepdrivewe agents should inherit from this class to ensure
    consistent behavior and integration with the Academy framework.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for this agent.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base Academy agent.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the parent Agent class.
        """
        super().__init__()

        # Set up logging for this agent
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f'Initialized {self.__class__.__name__}')

    def _log_action(self, action_name: str, **kwargs: Any) -> None:
        """Log an action invocation with parameters.

        Parameters
        ----------
        action_name : str
            Name of the action being invoked.
        **kwargs : Any
            Action parameters to log.
        """
        params_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        self.logger.debug(f'Action {action_name}({params_str})')

    def _log_error(
        self,
        action_name: str,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Log an error that occurred during action execution.

        Parameters
        ----------
        action_name : str
            Name of the action that failed.
        error : Exception
            The exception that was raised.
        **kwargs : Any
            Additional context to log.
        """
        context_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        self.logger.error(
            f'Error in {action_name}: {error!s} (context: {context_str})',
            exc_info=True,
        )

