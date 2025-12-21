"""
Download PDFs from arXiv using arXiv IDs.
This allows us to avoid storing PDFs in the repo and download them at runtime.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict


def download_arxiv_pdf(arxiv_id: str, output_dir: str = "data", force: bool = False) -> str:
    """
    Downloads a single PDF from arXiv.

    Args:
        arxiv_id: arXiv ID (e.g., "1706.03762")
        output_dir: Directory to save PDF
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded PDF
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find latest version
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{arxiv_id}.pdf"
    filepath = output_path / filename

    # Skip if already exists
    if filepath.exists() and not force:
        print(f"  ✓ {filename} (cached)")
        return str(filepath)

    try:
        print(f"  ⬇ Downloading {filename}...", end=" ", flush=True)
        response = requests.get(pdf_url, timeout=30, stream=True)
        response.raise_for_status()

        # Save PDF
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("✓")
        return str(filepath)

    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def download_all_papers(
    papers_file: str = "arxiv_papers.json",
    output_dir: str = "data",
    force: bool = False
) -> List[str]:
    """
    Download all papers listed in arxiv_papers.json.

    Args:
        papers_file: Path to JSON file with arXiv IDs
        output_dir: Directory to save PDFs
        force: If True, re-download all papers

    Returns:
        List of paths to downloaded PDFs
    """
    with open(papers_file, 'r') as f:
        data = json.load(f)

    papers = data.get("papers", [])
    print(f"\nDownloading {len(papers)} papers from arXiv...")
    print("=" * 60)

    downloaded = []
    failed = []

    for i, paper in enumerate(papers, 1):
        arxiv_id = paper["arxiv_id"]
        title = paper.get("title", "Unknown")

        print(f"\n[{i}/{len(papers)}] {title}")

        filepath = download_arxiv_pdf(arxiv_id, output_dir, force)
        if filepath:
            downloaded.append(filepath)
        else:
            failed.append(arxiv_id)

        # Rate limiting to be respectful to arXiv servers
        if i < len(papers):
            time.sleep(3)

    print("\n" + "=" * 60)
    print(f"✓ Downloaded: {len(downloaded)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for arxiv_id in failed:
            print(f"  - {arxiv_id}")

    return downloaded


def clean_downloaded_papers(output_dir: str = "data"):
    """
    Remove all downloaded PDFs (cleanup after processing).
    Useful for deployment where we only need vector embeddings.
    """
    output_path = Path(output_dir)
    pdf_files = list(output_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs to clean")
        return

    print(f"\nRemoving {len(pdf_files)} downloaded PDFs...")
    for pdf_file in pdf_files:
        pdf_file.unlink()
        print(f"  ✓ Removed {pdf_file.name}")

    print("Cleanup complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PDFs from arXiv")
    parser.add_argument("--papers-file", default="arxiv_papers.json",
                       help="JSON file with arXiv IDs")
    parser.add_argument("--output-dir", default="data",
                       help="Directory to save PDFs")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if files exist")
    parser.add_argument("--clean", action="store_true",
                       help="Remove all downloaded PDFs")

    args = parser.parse_args()

    if args.clean:
        clean_downloaded_papers(args.output_dir)
    else:
        download_all_papers(args.papers_file, args.output_dir, args.force)
