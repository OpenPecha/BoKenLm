# BoKenLm

BoKenLm is a project for training a KenLM n-gram language model for the Tibetan language. It uses SentencePiece for tokenization and KenLM for language model creation. This toolkit is designed to be straightforward for creating language models from a large text corpus.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd BoKenLm
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .env
    source .env/bin/activate
    ```

3.  **Install dependencies:**
    The project uses `pyproject.toml` to manage dependencies. Install them using pip:
    ```bash
    pip install -e .
    ```

## Usage

### Training the Language Model

The entire training process (SentencePiece tokenization and KenLM model building) is handled by a single script.

1.  **Prepare your data:**
    You need a large corpus of clean Tibetan text. The corpus should be in a single `.txt` file with one sentence per line.

2.  **Configure the training script:**
    Open the file `src/BoKenLm/train_lm.py` and modify the configuration section inside the `main()` function:
    ```python
    # Path to the clean Tibetan corpus file.
    corpus_path = "path/to/your/clean_corpus.txt"
    # Directory to save the trained models.
    output_dir = "models/kenlm"
    # Vocabulary size for SentencePiece tokenizer.
    vocab_size = 32000
    # N-gram order for the KenLM model.
    ngram = 5
    ```
    Update `corpus_path` to point to your text file. You can also adjust `output_dir`, `vocab_size`, and the `ngram` order.

3.  **Run training:**
    Execute the script from the root directory of the project:
    ```bash
    python src/BoKenLm/train_lm.py
    ```
    The script will first train a SentencePiece model and save it to your `output_dir`. Then, it will use that model to tokenize the corpus and train a KenLM model, saving the final `lm.arpa` file in the same directory.

### Tokenizing Text with the Trained Model

Once training is complete, you can use the generated SentencePiece model (`tokenizer.model` in your output directory) to tokenize new Tibetan text.

Here is an example Python snippet:

```python
import sentencepiece as spm

# Load the trained model
sp = spm.SentencePieceProcessor(model_file="models/kenlm/tokenizer.model")

# Example Tibetan text
tibetan_text = "བཀྲ་ཤིས་བདེ་ལེགས།"

# Encode text into tokens (pieces)
tokens = sp.encode_as_pieces(tibetan_text)
print(f"Tokens: {tokens}")

# Encode text into token IDs
ids = sp.encode_as_ids(tibetan_text)
print(f"Token IDs: {ids}")

# Decode from IDs back to text
decoded_text = sp.decode_ids(ids)
print(f"Decoded Text: {decoded_text}")
```

## Contributing

If you'd like to help out, check out our [contributing guidelines](/CONTRIBUTING.md).

## How to get help

* File an issue on the project's GitHub page.
* Email us at openpecha[at]gmail.com.
* Join our [discord](https://discord.com/invite/7GFpPFSTeA).

## License

BoKenLm is licensed under the [MIT License](/LICENSE.md).
