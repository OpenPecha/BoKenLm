"""
Example: Train a Tibetan KenLM language model and upload to Hugging Face.

Usage:
    python example.py
"""

from BoKenLm import (
    KenLMTrainer,
    SyllableTokenizer,
    SentencePieceTokenizer,
    HFUploader,
)

# Hugging Face organisation / user namespace
HF_NAMESPACE = "openpecha"


def train_with_syllable_tokenizer() -> KenLMTrainer:
    """Train using the syllable tokenizer (splits on tseg/shad)."""
    tokenizer = SyllableTokenizer()
    trainer = KenLMTrainer(
        tokenizer=tokenizer,
        corpus_path="data/bo_corpus.txt",
        output_dir="models/kenlm",
    )
    trainer.train()
    return trainer


def train_with_sentencepiece_tokenizer() -> KenLMTrainer:
    """Train using the BoSentencePiece tokenizer from HuggingFace."""
    tokenizer = SentencePieceTokenizer()
    trainer = KenLMTrainer(
        tokenizer=tokenizer,
        corpus_path="data/bo_corpus.txt",
        output_dir="models/kenlm",
    )
    trainer.train()
    return trainer


if __name__ == "__main__":
    # Switch between tokenizers by commenting/uncommenting:
    # trainer = train_with_syllable_tokenizer()
    trainer = train_with_sentencepiece_tokenizer()

    # Upload the trained model to Hugging Face
    uploader = HFUploader(
        repo_id=f"{HF_NAMESPACE}/{trainer._model_name}",
        model_dir=trainer.output_dir,
    )
    uploader.upload()
