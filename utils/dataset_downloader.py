# utils/dataset_downloader.py
import os
import hashlib
import tempfile
import tarfile
import zipfile
import urllib.request
import urllib.parse
from typing import Dict, Optional, Set
from pathlib import Path
import json
from .dataset_registry import DatasetInfo, DatasetFile


class DownloadError(Exception):
    """Exception raised for download-related errors."""
    pass


class DatasetDownloader:
    """Handles downloading and caching of datasets."""

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._download_cache_file = self.cache_dir / ".download_cache.json"
        self._download_cache = self._load_download_cache()

    def _load_download_cache(self) -> Dict[str, str]:
        """Load download cache metadata."""
        if self._download_cache_file.exists():
            try:
                with open(self._download_cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_download_cache(self) -> None:
        """Save download cache metadata."""
        try:
            with open(self._download_cache_file, 'w') as f:
                json.dump(self._download_cache, f, indent=2)
        except IOError:
            pass  # Non-critical error

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        if not expected_checksum:
            return True  # No checksum to verify

        actual_checksum = self._compute_file_hash(file_path)
        return actual_checksum.lower() == expected_checksum.lower()

    def _download_file(self, url: str, output_path: Path, description: str = "file") -> None:
        """Download a file with progress indication."""
        print(f"[download] Downloading {description} from {url}")

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 50 == 0 or percent >= 100:  # Print every ~50 blocks or at completion
                    print(f"[download] Progress: {percent}%", flush=True)

        try:
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, dir=output_path.parent) as tmp_file:
                tmp_path = Path(tmp_file.name)

            urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook)

            # Move to final location
            tmp_path.rename(output_path)
            print(f"[download] Completed: {output_path}")

        except Exception as e:
            # Clean up temporary file if it exists
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise DownloadError(f"Failed to download {url}: {e}")

    def _extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """Extract tar.gz or zip archive."""
        print(f"[extract] Extracting {archive_path}")

        try:
            if archive_path.suffix.lower() in {'.gz', '.tgz'} or '.tar.gz' in archive_path.name.lower():
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(extract_to)
            elif archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    zip_file.extractall(extract_to)
            else:
                raise DownloadError(f"Unsupported archive format: {archive_path}")

            print(f"[extract] Completed: extracted to {extract_to}")

        except Exception as e:
            raise DownloadError(f"Failed to extract {archive_path}: {e}")

    def _find_extracted_file(self, extract_dir: Path, target_filename: str) -> Optional[Path]:
        """Find a file in extracted directory (handles nested directory structures)."""
        # Try direct path first
        direct_path = extract_dir / target_filename
        if direct_path.exists():
            return direct_path

        # Search recursively
        for root, dirs, files in os.walk(extract_dir):
            if target_filename in files:
                return Path(root) / target_filename

        return None

    def download_dataset_file(self, dataset_file: DatasetFile, dataset_key: str) -> Path:
        """Download a single dataset file.

        Returns:
            Path: Local path to the downloaded/cached file
        """
        # Check if already cached and valid
        cache_key = f"{dataset_key}:{dataset_file.name}"
        final_path = self.cache_dir / f"{dataset_key}_{dataset_file.filename}"

        if cache_key in self._download_cache and final_path.exists():
            if not dataset_file.checksum or self._verify_checksum(final_path, dataset_file.checksum):
                print(f"[cache] Using cached file: {final_path}")
                return final_path
            else:
                print(f"[cache] Checksum mismatch for {final_path}, re-downloading")

        # Determine if we need to extract an archive
        parsed_url = urllib.parse.urlparse(dataset_file.url)
        url_filename = os.path.basename(parsed_url.path)
        is_archive = any(url_filename.lower().endswith(ext) for ext in ['.tar.gz', '.tgz', '.zip'])

        if is_archive:
            # Download archive first
            archive_path = self.cache_dir / f"{dataset_key}_{url_filename}"
            if not archive_path.exists():
                self._download_file(dataset_file.url, archive_path, f"{dataset_key} archive")

            # Extract archive
            extract_dir = self.cache_dir / f"{dataset_key}_extracted"
            if not extract_dir.exists():
                extract_dir.mkdir()
                self._extract_archive(archive_path, extract_dir)

            # Find the target file in extracted content
            extracted_file = self._find_extracted_file(extract_dir, dataset_file.filename)
            if not extracted_file:
                raise DownloadError(f"Could not find {dataset_file.filename} in extracted archive")

            # Copy to final location with standardized name
            if extracted_file != final_path:
                import shutil
                shutil.copy2(extracted_file, final_path)

        else:
            # Direct file download
            self._download_file(dataset_file.url, final_path, f"{dataset_key} {dataset_file.name}")

        # Verify checksum if provided
        if dataset_file.checksum:
            if not self._verify_checksum(final_path, dataset_file.checksum):
                final_path.unlink()  # Remove corrupted file
                raise DownloadError(f"Checksum verification failed for {final_path}")

        # Update cache
        self._download_cache[cache_key] = str(final_path)
        self._save_download_cache()

        return final_path

    def download_dataset(self, dataset_info: DatasetInfo) -> Dict[str, Path]:
        """Download all files for a dataset.

        Returns:
            Dict[str, Path]: Mapping from file component name to local path
        """
        print(f"[download] Downloading dataset: {dataset_info.name}")

        file_paths = {}
        for component_name, dataset_file in dataset_info.files.items():
            try:
                file_path = self.download_dataset_file(dataset_file, dataset_info.key)
                file_paths[component_name] = file_path
            except Exception as e:
                print(f"[download] Failed to download {component_name}: {e}")
                raise

        print(f"[download] Dataset {dataset_info.name} ready")
        return file_paths

    def is_dataset_cached(self, dataset_info: DatasetInfo) -> bool:
        """Check if all dataset files are already cached."""
        for component_name, dataset_file in dataset_info.files.items():
            cache_key = f"{dataset_info.key}:{dataset_file.name}"
            final_path = self.cache_dir / f"{dataset_info.key}_{dataset_file.filename}"

            if cache_key not in self._download_cache or not final_path.exists():
                return False

            # Verify checksum if available
            if dataset_file.checksum and not self._verify_checksum(final_path, dataset_file.checksum):
                return False

        return True

    def get_dataset_files(self, dataset_info: DatasetInfo) -> Dict[str, Path]:
        """Get local paths for dataset files (download if needed).

        Returns:
            Dict[str, Path]: Mapping from file component name to local path
        """
        if not self.is_dataset_cached(dataset_info):
            return self.download_dataset(dataset_info)

        # Return cached paths
        file_paths = {}
        for component_name, dataset_file in dataset_info.files.items():
            final_path = self.cache_dir / f"{dataset_info.key}_{dataset_file.filename}"
            file_paths[component_name] = final_path

        return file_paths

    def clear_cache(self, dataset_key: Optional[str] = None) -> None:
        """Clear download cache.

        Args:
            dataset_key: If provided, only clear cache for this dataset.
                        If None, clear entire cache.
        """
        if dataset_key is None:
            # Clear entire cache
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    file_path.unlink()
            self._download_cache.clear()
        else:
            # Clear specific dataset
            pattern = f"{dataset_key}_*"
            for file_path in self.cache_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()

            # Remove from cache metadata
            keys_to_remove = [k for k in self._download_cache.keys() if k.startswith(f"{dataset_key}:")]
            for key in keys_to_remove:
                del self._download_cache[key]

        self._save_download_cache()
        print(f"[cache] Cleared cache{f' for {dataset_key}' if dataset_key else ''}")


# Global downloader instance
_global_downloader: Optional[DatasetDownloader] = None


def get_downloader(cache_dir: str = "data") -> DatasetDownloader:
    """Get the global dataset downloader instance."""
    global _global_downloader
    if _global_downloader is None or _global_downloader.cache_dir != Path(cache_dir):
        _global_downloader = DatasetDownloader(cache_dir)
    return _global_downloader