
import os
import re
import subprocess
import sentencepiece as spm
from pathlib import Path
from huggingface_hub import hf_hub_download
from BoKenLm.upload_model import upload_to_huggingface

HF_TOKENIZER_REPO = "openpecha/BoSentencePiece"
SP_MODEL_FILENAME = "sentencepiece.model"


def get_sp_model_path():
    """Downloads the BoSentencePiece model from HuggingFace and returns the local path."""
    model_path = hf_hub_download(repo_id=HF_TOKENIZER_REPO, filename=SP_MODEL_FILENAME)
    return model_path


def parse_lmplz_log(log_text):
    """
    Parses lmplz stderr output and returns a dict of training statistics.
    """
    stats = {}

    # Unigram tokens and types
    m = re.search(r"Unigram tokens (\d+) types (\d+)", log_text)
    if m:
        stats["unigram_tokens"] = int(m.group(1))
        stats["unigram_types"] = int(m.group(2))

    # N-gram statistics: order, count, discount values
    ngram_stats = []
    for m in re.finditer(
        r"^(\d+)\s+(\d+)(?:/(\d+))?\s+D1=([\d.]+)\s+D2=([\d.]+)\s+D3\+=([\d.]+)",
        log_text,
        re.MULTILINE,
    ):
        ngram_stats.append({
            "order": int(m.group(1)),
            "count": int(m.group(2)),
            "D1": float(m.group(4)),
            "D2": float(m.group(5)),
            "D3": float(m.group(6)),
        })
    stats["ngram_stats"] = ngram_stats

    # Memory estimates
    mem_estimates = []
    for m in re.finditer(r"^(probing|trie)\s+(\d+)\s+(.+)$", log_text, re.MULTILINE):
        mem_estimates.append({
            "type": m.group(1),
            "mb": int(m.group(2)),
            "description": m.group(3).strip(),
        })
    stats["memory_estimates"] = mem_estimates

    # Resource usage from the Name:lmplz line
    m = re.search(
        r"Name:lmplz\s+VmPeak:(\d+)\s+kB\s+VmRSS:(\d+)\s+kB\s+"
        r"RSSMax:(\d+)\s+kB\s+user:([\d.]+)\s+sys:([\d.]+)\s+"
        r"CPU:([\d.]+)\s+real:([\d.]+)",
        log_text,
    )
    if m:
        stats["vm_peak_mb"] = round(int(m.group(1)) / 1024)
        stats["rss_max_mb"] = round(int(m.group(3)) / 1024)
        stats["user_time"] = float(m.group(4))
        stats["sys_time"] = float(m.group(5))
        stats["real_time"] = float(m.group(7))

    return stats


def generate_readme(output_dir, n_gram, corpus_path, lmplz_log):
    """
    Generates a README.md model card in the output directory from training stats.
    """
    stats = parse_lmplz_log(lmplz_log)
    corpus_name = Path(corpus_path).name

    lines = [
        "# BoKenLm - Tibetan KenLM Language Model",
        "",
        "A KenLM n-gram language model trained on Tibetan text, tokenized with "
        "[BoSentencePiece](https://huggingface.co/openpecha/BoSentencePiece).",
        "",
        "## Model Details",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| **Model Type** | Modified Kneser-Ney {n_gram}-gram |",
        "| **Tokenizer** | [openpecha/BoSentencePiece](https://huggingface.co/openpecha/BoSentencePiece) (Unigram, 20k vocab) |",
        f"| **Training Corpus** | `{corpus_name}` |",
        "| **Pruning** | 0 0 1 |",
    ]

    if "unigram_tokens" in stats:
        lines.append(f"| **Tokens** | {stats['unigram_tokens']:,} |")
        lines.append(f"| **Vocabulary Size** | {stats['unigram_types']:,} |")

    lines += ["", "## N-gram Statistics", ""]
    lines.append("| Order | Count | D1 | D2 | D3+ |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ng in stats.get("ngram_stats", []):
        lines.append(
            f"| {ng['order']} | {ng['count']:,} | {ng['D1']:.4f} | {ng['D2']:.4f} | {ng['D3']:.4f} |"
        )

    if stats.get("memory_estimates"):
        lines += ["", "## Memory Estimates", ""]
        lines.append("| Type | MB | Details |")
        lines.append("| --- | --- | --- |")
        for mem in stats["memory_estimates"]:
            lines.append(f"| {mem['type']} | {mem['mb']} | {mem['description']} |")

    if "real_time" in stats:
        lines += ["", "## Training Resources", ""]
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| **Peak Virtual Memory** | {stats['vm_peak_mb']:,} MB |")
        lines.append(f"| **Peak RSS** | {stats['rss_max_mb']:,} MB |")
        lines.append(f"| **Wall Time** | {stats['real_time']:.1f}s |")
        lines.append(f"| **User Time** | {stats['user_time']:.1f}s |")
        lines.append(f"| **System Time** | {stats['sys_time']:.1f}s |")

    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "import kenlm",
        "",
        'model = kenlm.Model("lm.arpa")',
        "",
        "# Score a tokenized sentence",
        'score = model.score("▁བོད་སྐད་ ▁ཀྱི་ ▁ཚིག་གྲུབ་ ▁འདི་ ▁ཡིན།")',
        "print(score)",
        "```",
        "",
        "## Files",
        "",
        "- `lm.arpa` — ARPA format language model",
        "- `README.md` — This model card",
        "",
        "## License",
        "",
        "Apache 2.0",
        "",
    ]

    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text("\n".join(lines))
    print(f"README.md saved to {readme_path}")


def train_kenlm(corpus_path, arpa_path, output_dir, n_gram=5):
    """
    Trains a KenLM n-gram model.

    Args:
        corpus_path (str): Path to the training corpus.
        arpa_path (str): Path to save the output ARPA model.
        output_dir (str): Directory to save model files and README.
        n_gram (int): The 'n' for n-gram model.
    """
    print("Downloading BoSentencePiece tokenizer from HuggingFace...")
    sp_model_path = get_sp_model_path()
    print("Tokenizing corpus with SentencePiece...")
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    tokenized_corpus = ''
    tokenized_path = Path('./data/tokenized_corpus.txt')
    lines = Path(corpus_path).read_text().splitlines()
    for line in lines:
        tokens = sp.encode_as_pieces(line.strip())
        tokenized_corpus += " ".join(tokens) + "\n"
    tokenized_path.write_text(tokenized_corpus)
    print("Training KenLM model...")
    kenlm_bin_dir = os.path.expanduser('~/.local/bin')
    lmplz_path = os.path.join(kenlm_bin_dir, 'lmplz')
    
    if not os.path.exists(lmplz_path):
        print("\nERROR: `lmplz` not found.")
        print("Please ensure KenLM is installed and `lmplz` is in your PATH.")
        print(f"Searched in: {kenlm_bin_dir}")
        print("\nTo install KenLM from source:")
        print("  git clone https://github.com/kpu/kenlm.git /tmp/kenlm")
        print("  cd /tmp/kenlm && mkdir build && cd build")
        print("  cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local")
        print("  cmake --build . -j$(nproc)")
        print("  cmake --install .")
        print("\nOr check its location with: find / -name lmplz 2>/dev/null\n")
        exit(1)

    # Ensure Boost shared libs are found at runtime
    local_lib = os.path.expanduser('~/.local/lib')
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = local_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    command = [
        lmplz_path,
        "-o", str(n_gram),
        "--text", str(tokenized_path),
        "--arpa", arpa_path,
        "--prune", "0", "0", "1" # Pruning for perplexity models
    ]
    
    # Capture stderr to parse training stats for README
    result = subprocess.run(command, check=True, env=env, stderr=subprocess.PIPE, text=True)
    lmplz_log = result.stderr
    print(lmplz_log)
    
    # Clean up the temporary tokenized file
    os.remove(tokenized_path)
    
    print(f"KenLM model saved to {arpa_path}")

    # Generate README model card from training stats
    generate_readme(output_dir, n_gram, corpus_path, lmplz_log)


def main():
    # --- Configuration ---
    # Path to the clean Tibetan corpus file.
    corpus_path = "data/bo_corpus.txt"
    # Directory to save the trained models.
    output_dir = "models/kenlm"
    # N-gram order for the KenLM model.
    ngram = 5
    # Hugging Face repo ID to upload to (set to None to skip upload)
    hf_repo_id = "openpecha/BoKenLm"  # e.g. "openpecha/BoKenLm"
    # --- End of Configuration ---

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    arpa_path = os.path.join(output_dir, "lm.arpa")
    train_kenlm(corpus_path, arpa_path, output_dir, ngram)

    print("\nTraining complete.")
    print(f"Models are saved in: {output_dir}")

    # Upload to Hugging Face if repo_id is set
    if hf_repo_id:
        upload_to_huggingface(repo_id=hf_repo_id, model_dir=output_dir)

if __name__ == "__main__":
    main()
