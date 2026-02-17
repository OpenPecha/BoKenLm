
import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


DEFAULT_MODEL_DIR = "models/kenlm"


def upload_to_huggingface(repo_id, model_dir=DEFAULT_MODEL_DIR, private=False):
    """
    Uploads the trained KenLM model files to a Hugging Face repository.

    Args:
        repo_id (str): The Hugging Face repo ID (e.g. 'openpecha/BoKenLm').
        model_dir (str): Local directory containing the trained KenLM model files.
        private (bool): Whether the repo should be private.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_files = list(model_dir.glob("*"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in: {model_dir}")

    api = HfApi()

    # Create repo if it doesn't exist (no-op if it already exists)
    print(f"Creating/verifying repo: {repo_id} ...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    # Upload all files in the model directory
    print(f"Uploading {len(model_files)} file(s) from '{model_dir}' to '{repo_id}' ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload KenLM model",
    )

    print(f"Upload complete! View your model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained KenLM model to Hugging Face."
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repo ID (e.g. 'openpecha/BoKenLm').",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing the KenLM model files (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face repo private.",
    )
    args = parser.parse_args()

    upload_to_huggingface(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        private=args.private,
    )


if __name__ == "__main__":
    main()
