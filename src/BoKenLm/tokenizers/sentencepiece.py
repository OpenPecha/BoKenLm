import sentencepiece as spm
from huggingface_hub import hf_hub_download

from BoKenLm.tokenizers.base import BaseTokenizer

HF_TOKENIZER_REPO = "openpecha/BoSentencePiece"
SP_MODEL_FILENAME = "sentencepiece.model"


class SentencePieceTokenizer(BaseTokenizer):
    """
    Tokenizes Tibetan text using the BoSentencePiece model from HuggingFace.

    The model is downloaded automatically on first use and cached locally.

    Example:
        >>> tok = SentencePieceTokenizer()
        >>> tokens = tok.tokenize("བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།")
    """

    def __init__(self):
        print("Downloading BoSentencePiece tokenizer from HuggingFace...")
        model_path = hf_hub_download(
            repo_id=HF_TOKENIZER_REPO, filename=SP_MODEL_FILENAME
        )
        self._sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def name(self) -> str:
        return "sentencepiece"

    @property
    def description(self) -> str:
        return (
            "[openpecha/BoSentencePiece](https://huggingface.co/openpecha/BoSentencePiece) "
            "(Unigram, 20k vocab)"
        )

    def tokenize(self, text: str) -> list[str]:
        return self._sp.encode_as_pieces(text)
