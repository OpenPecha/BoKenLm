from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for config and README (e.g. 'syllable')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the README model card."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize a single line of text.

        Args:
            text: A line of text to tokenize.

        Returns:
            A list of token strings.
        """
        ...
