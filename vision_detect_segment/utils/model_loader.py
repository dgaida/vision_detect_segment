import hashlib
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Predefined checksums for known models
MODEL_CHECKSUMS: Dict[str, str] = {
    "yoloe-11s-seg.pt": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", # Example
}

def verify_model_checksum(model_path: Path, expected_checksum_str: str) -> bool:
    """
    Verify model file checksum.

    Args:
        model_path: Path to the model file
        expected_checksum_str: Checksum string in format "algo:hash"

    Returns:
        True if checksum matches, False otherwise
    """
    if ":" not in expected_checksum_str:
        logger.warning(f"Invalid checksum format: {expected_checksum_str}")
        return False

    algorithm, expected_hash = expected_checksum_str.split(":", 1)

    try:
        hasher = hashlib.new(algorithm)
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        actual_hash = hasher.hexdigest()
        return actual_hash == expected_hash
    except Exception as e:
        logger.error(f"Error calculating checksum: {e}")
        return False

def get_model_path_safe(model_id: str, model_path: str) -> str:
    """
    Get model path and verify checksum if available.

    Args:
        model_id: Model identifier
        model_path: Path to the model file

    Returns:
        Verified model path

    Raises:
        RuntimeError: If checksum verification fails
    """
    path = Path(model_path)

    # If the file exists and we have a checksum for it, verify it
    filename = path.name
    if path.exists() and filename in MODEL_CHECKSUMS:
        if not verify_model_checksum(path, MODEL_CHECKSUMS[filename]):
            raise RuntimeError(f"Model checksum mismatch for {filename}")
        logger.info(f"Model checksum verified for {filename}")

    return str(path)
