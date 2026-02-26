from pathlib import Path

from huggingface_hub import HfApi, create_repo


class HFUploader:
    """Uploads trained KenLM model files to a Hugging Face repository.

    Args:
        repo_id: The Hugging Face repo ID (e.g. ``openpecha/BoKenlm-syl``).
        model_dir: Local directory containing the trained model files.
        private: Whether the HF repo should be private.

    Example:
        >>> from BoKenLm import HFUploader
        >>> uploader = HFUploader(
        ...     repo_id="openpecha/BoKenlm-syl",
        ...     model_dir="models/kenlm",
        ... )
        >>> uploader.upload()
    """

    def __init__(
        self,
        repo_id: str,
        model_dir: str = "models/kenlm",
        private: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.model_dir = Path(model_dir)
        self.private = private
        self._api = HfApi()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(self) -> None:
        """Create the remote repo (if needed) and upload all model files."""
        self._validate_model_dir()
        self._create_repo()
        self._upload_files()
        print(
            f"Upload complete! View your model at: "
            f"https://huggingface.co/{self.repo_id}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_model_dir(self) -> None:
        """Ensure the model directory exists and is not empty."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )
        if not any(self.model_dir.iterdir()):
            raise FileNotFoundError(
                f"No model files found in: {self.model_dir}"
            )

    def _create_repo(self) -> None:
        """Create the HF repo if it does not already exist."""
        print(f"Creating/verifying repo: {self.repo_id} ...")
        create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            exist_ok=True,
            private=self.private,
        )

    def _upload_files(self) -> None:
        """Upload the entire model directory to the HF repo."""
        file_count = sum(1 for _ in self.model_dir.iterdir())
        print(
            f"Uploading {file_count} file(s) from "
            f"'{self.model_dir}' to '{self.repo_id}' ..."
        )
        self._api.upload_folder(
            folder_path=str(self.model_dir),
            repo_id=self.repo_id,
            repo_type="model",
            commit_message="Upload KenLM model",
        )
