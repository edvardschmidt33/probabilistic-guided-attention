import os
import argparse
from tqdm import tqdm
from huggingface_hub import HfFileSystem, hf_hub_download

DATASET_REPO = "timm/imagenet-1k-wds"
SPLIT_PATTERN = "**/*-validation-*.tar"  # Matches validation tar files

def main(download_dir: str):
    """
    Download all validation tar files for the ImageNet dataset from the Hugging Face Datasets Hub.

    Args:
        download_dir (str): Directory to download validation tar files
    """
    # Initialize Hugging Face filesystem
    fs = HfFileSystem()

    # Get list of validation tar files
    files = [fs.resolve_path(path) for path in fs.glob(f"hf://datasets/{DATASET_REPO}/{SPLIT_PATTERN}")]

    # Create local directory if it doesn’t exist
    os.makedirs(download_dir, exist_ok=True)

    # Download each tar file with tqdm progress bar
    for file in tqdm(files, desc="Downloading validation tar files", unit="file"):
        hf_hub_download(
            repo_id=file.repo_id,
            filename=file.path_in_repo,
            repo_type="dataset",
            local_dir=download_dir
        )

    print(f"\n✅ All {len(files)} validation tar files downloaded to '{download_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, required=True, help="Directory to download validation tar files")
    args = parser.parse_args()
    main(args.download_dir)