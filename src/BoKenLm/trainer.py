import os
import re
import subprocess
from pathlib import Path

from botok.utils.corpus_normalization import normalize_corpus
from tqdm import tqdm

from BoKenLm.tokenizers.base import BaseTokenizer


class KenLMTrainer:
    """
    Trains a KenLM n-gram language model using a pluggable tokenizer.

    The output ARPA file is named after the tokenizer:
      - SentencePiece → ``BoKenlm-sp.arpa``
      - Syllable      → ``BoKenlm-syl.arpa``

    Args:
        tokenizer: A BaseTokenizer instance for tokenizing the corpus.
        corpus_path: Path to the training corpus text file.
        output_dir: Directory to save the ARPA model and README.
        n_gram: The n-gram order for the KenLM model.

    Example:
        >>> from BoKenLm import KenLMTrainer, SyllableTokenizer
        >>> trainer = KenLMTrainer(
        ...     tokenizer=SyllableTokenizer(),
        ...     corpus_path="data/bo_corpus.txt",
        ...     output_dir="models/kenlm",
        ... )
        >>> trainer.train()
    """

    # Maps tokenizer name → short suffix used in the model filename.
    _TOKENIZER_SUFFIXES: dict[str, str] = {
        "sentencepiece": "sp",
        "syllable": "syl",
    }

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        corpus_path: str,
        output_dir: str = "models/kenlm",
        n_gram: int = 5,
    ):
        self.tokenizer = tokenizer
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        self.n_gram = n_gram

        self._tokenized_path = Path("./data/tokenized_corpus.txt")
        self._model_name = self._derive_model_name(tokenizer.name)
        self._arpa_path = os.path.join(output_dir, f"{self._model_name}.arpa")

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------

    @classmethod
    def _derive_model_name(cls, tokenizer_name: str) -> str:
        """Return the model name (e.g. 'BoKenlm-syl') for a given tokenizer name."""
        suffix = cls._TOKENIZER_SUFFIXES.get(tokenizer_name)
        if suffix is None:
            raise ValueError(
                f"Unknown tokenizer '{tokenizer_name}'. "
                f"Supported: {list(cls._TOKENIZER_SUFFIXES)}"
            )
        return f"BoKenlm-{suffix}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training pipeline: tokenize (with normalization), train lmplz, generate README."""
        os.makedirs(self.output_dir, exist_ok=True)

        self._tokenize_corpus()
        lmplz_log = self._run_lmplz()

        # Clean up temporary tokenized file
        self._tokenized_path.unlink(missing_ok=True)

        print(f"KenLM model saved to {self._arpa_path}")
        self._generate_readme(lmplz_log)
        print(f"\nTraining complete. Models are saved in: {self.output_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_corpus(self) -> None:
        """Normalize and tokenize the corpus line-by-line, then write a space-separated file."""
        print(
            f"Normalizing & tokenizing corpus with "
            f"{self.tokenizer.name} tokenizer..."
        )
        tokenized_lines: list[str] = []
        corpus_lines = Path(self.corpus_path).read_text().splitlines()
        for line in tqdm(corpus_lines, desc="Normalizing & tokenizing"):
            normalized_line = normalize_corpus(line.strip())
            tokens = self.tokenizer.tokenize(normalized_line)
            tokenized_lines.append(" ".join(tokens))
        self._tokenized_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenized_path.write_text("\n".join(tokenized_lines) + "\n")

    def _run_lmplz(self) -> str:
        """Invoke lmplz and return its stderr log."""
        print("Training KenLM model...")
        kenlm_bin_dir = os.path.expanduser("~/.local/bin")
        lmplz_path = os.path.join(kenlm_bin_dir, "lmplz")

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
            raise FileNotFoundError(f"lmplz not found in {kenlm_bin_dir}")

        # Ensure Boost shared libs are found at runtime
        local_lib = os.path.expanduser("~/.local/lib")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = local_lib + ":" + env.get("LD_LIBRARY_PATH", "")

        command = [
            lmplz_path,
            "-o", str(self.n_gram),
            "--text", str(self._tokenized_path),
            "--arpa", self._arpa_path,
            "--prune", "0", "0", "1",
        ]

        result = subprocess.run(
            command, check=True, env=env, stderr=subprocess.PIPE, text=True
        )
        lmplz_log = result.stderr
        print(lmplz_log)
        return lmplz_log

    # ------------------------------------------------------------------
    # README generation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_lmplz_log(log_text: str) -> dict:
        """Parse lmplz stderr output into a stats dictionary."""
        stats: dict = {}

        m = re.search(r"Unigram tokens (\d+) types (\d+)", log_text)
        if m:
            stats["unigram_tokens"] = int(m.group(1))
            stats["unigram_types"] = int(m.group(2))

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

        mem_estimates = []
        for m in re.finditer(
            r"^(probing|trie)\s+(\d+)\s+(.+)$", log_text, re.MULTILINE
        ):
            mem_estimates.append({
                "type": m.group(1),
                "mb": int(m.group(2)),
                "description": m.group(3).strip(),
            })
        stats["memory_estimates"] = mem_estimates

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

    def _generate_readme(self, lmplz_log: str):
        """Generate a README.md model card from training stats."""
        stats = self._parse_lmplz_log(lmplz_log)
        corpus_name = Path(self.corpus_path).name
        tok = self.tokenizer

        lines = [
            f"# {self._model_name} - Tibetan KenLM Language Model",
            "",
            f"A KenLM n-gram language model trained on Tibetan text, "
            f"tokenized with {tok.name} tokenizer.",
            "",
            "## Model Details",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| **Model Type** | Modified Kneser-Ney {self.n_gram}-gram |",
            f"| **Tokenizer** | {tok.description} |",
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
                f"| {ng['order']} | {ng['count']:,} "
                f"| {ng['D1']:.4f} | {ng['D2']:.4f} | {ng['D3']:.4f} |"
            )

        if stats.get("memory_estimates"):
            lines += ["", "## Memory Estimates", ""]
            lines.append("| Type | MB | Details |")
            lines.append("| --- | --- | --- |")
            for mem in stats["memory_estimates"]:
                lines.append(
                    f"| {mem['type']} | {mem['mb']} | {mem['description']} |"
                )

        if "real_time" in stats:
            lines += ["", "## Training Resources", ""]
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            lines.append(f"| **Peak Virtual Memory** | {stats['vm_peak_mb']:,} MB |")
            lines.append(f"| **Peak RSS** | {stats['rss_max_mb']:,} MB |")
            lines.append(f"| **Wall Time** | {stats['real_time']:.1f}s |")
            lines.append(f"| **User Time** | {stats['user_time']:.1f}s |")
            lines.append(f"| **System Time** | {stats['sys_time']:.1f}s |")

        if tok.name == "sentencepiece":
            usage_example = (
                'score = model.score("▁བོད་སྐད་ ▁ཀྱི་ ▁ཚིག་གྲུབ་ ▁འདི་ ▁ཡིན།")'
            )
        else:
            usage_example = (
                'score = model.score("བོད་ སྐད་ ཀྱི་ ཚིག་ གྲུབ་ འདི་ ཡིན།")'
            )

        arpa_filename = f"{self._model_name}.arpa"
        lines += [
            "",
            "## Usage",
            "",
            "```python",
            "import kenlm",
            "",
            f'model = kenlm.Model("{arpa_filename}")',
            "",
            "# Score a tokenized sentence",
            usage_example,
            "print(score)",
            "```",
            "",
            "## Files",
            "",
            f"- `{arpa_filename}` — ARPA format language model",
            "- `README.md` — This model card",
            "",
            "## License",
            "",
            "Apache 2.0",
            "",
        ]

        readme_path = Path(self.output_dir) / "README.md"
        readme_path.write_text("\n".join(lines))
        print(f"README.md saved to {readme_path}")
