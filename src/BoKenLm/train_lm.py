import argparse
import os
import subprocess
import sentencepiece as spm

from .train_sentence_piece import train_sentencepiece


def train_kenlm(corpus_path, sp_model_path, arpa_path, n_gram=5):
    """
    Trains a KenLM n-gram model.

    Args:
        corpus_path (str): Path to the training corpus.
        sp_model_path (str): Path to the trained SentencePiece model.
        arpa_path (str): Path to save the output ARPA model.
        n_gram (int): The 'n' for n-gram model.
    """
    print("Tokenizing corpus with SentencePiece...")
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    tokenized_path = "tokenized_corpus.txt"
    with open(corpus_path, 'r', encoding='utf-8') as infile, \
         open(tokenized_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            tokens = sp.encode_as_pieces(line.strip())
            outfile.write(" ".join(tokens) + "\n")
    
    print("Training KenLM model...")
    kenlm_path = os.path.expanduser('~/.local/bin') # Common install path for pip user installs
    lmplz_path = os.path.join(kenlm_path, 'lmplz')
    
    if not os.path.exists(lmplz_path):
        print("\nERROR: `lmplz` not found.")
        print("Please ensure KenLM is installed and `lmplz` is in your PATH.")
        print(f"Searched in: {kenlm_path}")
        print("You can install it via: pip install kenlm-wheel")
        print("Or check its location with: find / -name lmplz 2>/dev/null\n")
        # Clean up the temporary tokenized file before exiting
        os.remove(tokenized_path)
        exit(1)

    command = [
        lmplz_path,
        "-o", str(n_gram),
        "--text", tokenized_path,
        "--arpa", arpa_path,
        "--prune", "0", "0", "1" # Pruning for perplexity models
    ]
    
    subprocess.run(command, check=True)
    
    # Clean up the temporary tokenized file
    os.remove(tokenized_path)
    
    print(f"KenLM model saved to {arpa_path}")

def main():
    # --- Configuration ---
    # Path to the clean Tibetan corpus file.
    corpus_path = "path/to/your/clean_corpus.txt"
    # Directory to save the trained models.
    output_dir = "models/kenlm"
    # Vocabulary size for SentencePiece tokenizer.
    vocab_size = 32000
    # N-gram order for the KenLM model.
    ngram = 5
    # --- End of Configuration ---

    os.makedirs(output_dir, exist_ok=True)

    sp_model_prefix = os.path.join(output_dir, "tokenizer")
    train_sentencepiece(corpus_path, sp_model_prefix, vocab_size)

    sp_model_path = f"{sp_model_prefix}.model"
    arpa_path = os.path.join(output_dir, "lm.arpa")
    train_kenlm(corpus_path, sp_model_path, arpa_path, ngram)

    print("\nTraining complete.")
    print(f"Models are saved in: {output_dir}")

if __name__ == "__main__":
    main()
