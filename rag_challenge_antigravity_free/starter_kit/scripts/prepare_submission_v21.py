import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Constants
ROOT_DIR = Path(__file__).resolve().parents[1]

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
    count = 0
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
            count += 1
    print(f"Archived {count} files.")
    return archive_path

def main():
    submission_path = ROOT_DIR / "submission.json"
    archive_path = ROOT_DIR / "code_archive.zip"
    
    if not submission_path.exists():
        print(f"CRITICAL: {submission_path} not found! Please ensure you have the generation phase complete.")
        return

    print(f"Generating clean code archive at {archive_path}...")
    ensure_code_archive(archive_path)
    
    print("\nPreparation complete!")
    print(f"1. Submission file: {submission_path} ({os.path.getsize(submission_path) // 1024} KB)")
    print(f"2. Code archive:   {archive_path} ({os.path.getsize(archive_path) // 1024} KB)")
    print("\nThese files are ready for manual submission.")

if __name__ == "__main__":
    main()
