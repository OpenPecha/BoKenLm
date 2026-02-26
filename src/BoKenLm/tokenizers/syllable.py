import re

from BoKenLm.tokenizers.base import BaseTokenizer


class SyllableTokenizer(BaseTokenizer):
    """
    Tokenizes Tibetan text into syllables, keeping tseg (་) and shad (།)
    attached to the preceding syllable.

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
        return "Tibetan syllable-based (split on tseg `་` / shad `།`)"

    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[^་།]+[་།]?", text)
        return [t.strip() for t in tokens if t.strip()]
