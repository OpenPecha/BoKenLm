from botok_rs import SimpleTokenizer as BotokTokenizer

from BoKenLm.tokenizers.base import BaseTokenizer


class SyllableTokenizer(BaseTokenizer):
    """Tokenizes Tibetan text into syllables using botok-rs.

    Example:
        >>> tok = SyllableTokenizer()
        >>> tok.tokenize("བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།")
        ['བོད་', 'སྐད་', 'ཀྱི་', 'ཚིག་', 'གྲུབ་', 'འདི་', 'ཡིན།']
    """

    @property
    def name(self) -> str:
        return "syllable"

    @property
    def description(self) -> str:
        return "Tibetan syllable-based (botok-rs SimpleTokenizer)"

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single line of Tibetan text into syllables.

        Args:
            text: A line of Tibetan text to tokenize.

        Returns:
            A list of syllable strings.
        """
        tokens = BotokTokenizer.tokenize(text)
        return [token.text for token in tokens if token.text.strip()]
