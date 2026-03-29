import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Constants
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import EvaluationClient, get_config

CONFIG = get_config()

_EXCLUDE_DIRS = {
    "__pycache__", "docs_corpus", "storage", ".venv", "venv", "env", 
    "document_indices", "code_archive_v1", "code_archive_v4", "leaderboard"
}
_EXCLUDE_FILES = {
    ".env", "submission.json", "questions.json", "code_archive.zip", 
    "submission_v21_multi_step.json", "document_metadata.json", "document_metadata.csv"
}
_EXCLUDE_PATTERNS = ["submission_v20_", "submission_comparison_", "code_archive_"]

def ensure_code_archive(archive_path: Path) -> Path:
    """Archive the entire starter_kit directory, excluding generated/runtime artifacts."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for file_path in ROOT_DIR.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() == archive_resolved:
                continue
            
            # Get relative path parts
            try:
                rel_path = file_path.relative_to(ROOT_DIR)
                parts = set(rel_path.parts)
            except ValueError:
                continue
                
            if parts & _EXCLUDE_DIRS:
                continue
            if file_path.name in _EXCLUDE_FILES:
                continue
            
            # Pattern matching for files and directories
            skip = False
            for pattern in _EXCLUDE_PATTERNS:
                if pattern in str(rel_path):
                    skip = True
                    break
            if skip:
                continue
            
            if file_path.suffix in [".zip", ".csv"]:
                continue

            zip_file.write(file_path, rel_path)
    return archive_path

def main():
    client = EvaluationClient.from_env()
    submission_path = ROOT_DIR / "submission.json"
    archive_path = ROOT_DIR / "code_archive.zip"
    
    print(f"Creating code archive at {archive_path}...")
    ensure_code_archive(archive_path)
    
    print(f"Submitting {submission_path}...")
    try:
        response = client.submit_submission(
            submission_path=submission_path,
            code_archive_path=archive_path
        )
        print("Submission successful!")
        print(json.dumps(response, indent=2))
        
        # Poll for results
        sub_uuid = response.get("submission_uuid")
        if sub_uuid:
            print(f"Waiting for results for submission {sub_uuid}...")
            # We can't easily poll here without a loop, but the API might return immediate status if processed fast.
            # In a real environment, we'd poll client.get_submission_status(sub_uuid).
    except Exception as e:
        print(f"Submission failed: {e}")

if __name__ == "__main__":
    import json
    main()
