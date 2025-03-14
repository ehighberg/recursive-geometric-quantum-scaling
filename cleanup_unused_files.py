"""
Script to clean up unused files and move them to the archive directory.
"""

import os
import shutil
from pathlib import Path
import datetime

# Define the archive directory
ARCHIVE_DIR = Path("archive")

# Create archive subdirectories if they don't exist
ARCHIVE_DIRS = {
    "docs": ARCHIVE_DIR / "docs",
    "reports": ARCHIVE_DIR / "reports",
    "images": ARCHIVE_DIR / "images",
    "backups": ARCHIVE_DIR / "backups",
    "old_scripts": ARCHIVE_DIR / "old_scripts",
    "old_tests": ARCHIVE_DIR / "old_tests",
    "cleanup": ARCHIVE_DIR / "cleanup"
}

for dir_path in ARCHIVE_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Files to move to archive
FILES_TO_ARCHIVE = {
    # Old scripts
    "run_phi_resonant_analysis_consolidated.py": ARCHIVE_DIRS["old_scripts"],
    "enhanced_phi_analysis_fixed.py": ARCHIVE_DIRS["old_scripts"],
    "create_table_image_fixed.py": ARCHIVE_DIRS["old_scripts"],
    "run_test.py": ARCHIVE_DIRS["old_scripts"],
    "test_fixed_implementations.py": ARCHIVE_DIRS["old_scripts"],
    "analyses/scaling/analyze_fs_scaling.py": ARCHIVE_DIRS["old_scripts"],
    "analyses/scaling/analyze_phi_significance.py": ARCHIVE_DIRS["old_scripts"],
    "analyses/scaling/analyze_fractal_topology_relation.py": ARCHIVE_DIRS["old_scripts"],
    "analyses/fractal_analysis_fixed_complete.py": ARCHIVE_DIRS["old_scripts"],
    "cleanup_old_files.py": ARCHIVE_DIRS["cleanup"],
    
    # Documentation files
    "audit_plan.md": ARCHIVE_DIRS["docs"],
    "PAPER_GUIDE.md": ARCHIVE_DIRS["docs"],
    
    # Backup files
    "simulations/scaled_unitary_fixed.py": ARCHIVE_DIRS["backups"],
}

# Directories to move to archive
DIRS_TO_ARCHIVE = {
    "test_results": ARCHIVE_DIRS["old_tests"],
    "test_results_phi": ARCHIVE_DIRS["old_tests"],
    "test_results_phi_separate": ARCHIVE_DIRS["old_tests"],
    "test_results_fixed": ARCHIVE_DIRS["old_tests"],
    "file_backups_20250309_182352": ARCHIVE_DIRS["backups"],
    "file_backups_20250309_182431": ARCHIVE_DIRS["backups"],
}

def archive_files():
    """Move files to archive directory."""
    print("Archiving files...")
    
    # Archive individual files
    for file_path, archive_dir in FILES_TO_ARCHIVE.items():
        src = Path(file_path)
        if src.exists():
            dst = archive_dir / src.name
            print(f"Moving {src} to {dst}")
            shutil.move(str(src), str(dst))
        else:
            print(f"File not found: {src}")
    
    # Archive directories
    for dir_path, archive_dir in DIRS_TO_ARCHIVE.items():
        src = Path(dir_path)
        if src.exists():
            dst = archive_dir / src.name
            print(f"Moving directory {src} to {dst}")
            shutil.move(str(src), str(dst))
        else:
            print(f"Directory not found: {src}")

def main():
    """Main function."""
    print(f"Starting cleanup at {datetime.datetime.now()}")
    
    # Create archive directory if it doesn't exist
    ARCHIVE_DIR.mkdir(exist_ok=True)
    
    # Archive files
    archive_files()
    
    print(f"Cleanup completed at {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
