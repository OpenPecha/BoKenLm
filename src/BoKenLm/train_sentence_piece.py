
import sentencepiece as spm
import subprocess

def train_sentencepiece(corpus_path, model_prefix, vocab_size):
    """
    Trains a SentencePiece model.

    Args:
        corpus_path (str): Path to the training corpus.
        model_prefix (str): Prefix for the model and vocab files.
        vocab_size (int): The size of the vocabulary.
    """
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        f'--input={corpus_path} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} --model_type=bpe'
    )
    print(f"SentencePiece model saved to {model_prefix}.model")

if __name__ == "__main__":
    corpus_path = "./data/clean_corpus.txt"
    model_prefix = "Bo_sentencepiece"
    vocab_size = 32000
    train_sentencepiece(corpus_path, model_prefix, vocab_size)