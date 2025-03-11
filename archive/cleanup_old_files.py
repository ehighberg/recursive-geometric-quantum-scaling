#!/usr/bin/env python
"""
Script to delete graph and table files older than 2 days.
Targets PNG, CSV, HTML, TEX, and JSON files in specific directories.
"""

import os
import time
import datetime
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Clean up old graph and table files.')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    parser.add_argument('--backup', action='store_true', help='Create backups before deletion')
    parser.add_argument('--days', type=float, default=2.0, help='Age threshold in days (default: 2.0)')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    # Calculate the cutoff time
    cutoff_time = time.time() - (args.days * 24 * 60 * 60)
    cutoff_date = datetime.datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Looking for files older than {args.days} days (before {cutoff_date})")

    # Directories to check
    directories = [
        "plots",
        "report",
        "data", 
        "validation_results",
        "test_results",
        "test_results_complete",
        "test_results_fixed",
        "test_results_phi",
        "test_results_phi_separate"
    ]

    # File extensions to target
    target_extensions = [
        ".png", ".jpg", ".jpeg", ".gif",  # Images
        ".csv", ".html", ".tex",          # Tables
        ".json"                           # Data files
    ]

    # Track results
    old_files = []
    
    # Find old files
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Directory not found: {directory}")
            continue
            
        print(f"Scanning directory: {directory}")
        for file_path in dir_path.glob('**/*'):
            if not file_path.is_file():
                continue
                
            # Check if extension matches our targets
            if file_path.suffix.lower() not in target_extensions:
                continue
                
            # Check file age
            try:
                mod_time = file_path.stat().st_mtime
                if mod_time < cutoff_time:
                    file_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    old_files.append((file_path, file_date))
            except Exception as e:
                print(f"Error checking file {file_path}: {e}")
    
    # Report findings
    if not old_files:
        print("\nNo files found older than the threshold.")
        return
    
    print(f"\nFound {len(old_files)} files older than {args.days} days:")
    for file_path, file_date in old_files:
        print(f"- {file_path} (Last modified: {file_date})")
    
    # Exit if dry run
    if args.dry_run:
        print("\nDRY RUN MODE: No files were deleted.")
        return
    
    # Confirmation
    if not args.no_confirm:
        confirmation = input(f"\nAre you sure you want to delete these {len(old_files)} files? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Create backup directory if needed
    if args.backup:
        backup_dir = Path("file_backups_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        backup_dir.mkdir(exist_ok=True)
        print(f"Creating backups in: {backup_dir}")
    
    # Process deletions
    deleted_files = []
    error_files = []
    
    for file_path, _ in old_files:
        try:
            # Backup if requested
            if args.backup:
                import shutil
                # Create a safe backup path that preserves directory structure
                backup_path = backup_dir / str(file_path).replace(':', '').replace('\\', '_').replace('/', '_')
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                print(f"Backed up: {file_path} → {backup_path}")
            
            # Delete the file
            file_path.unlink()
            deleted_files.append(str(file_path))
            print(f"Deleted: {file_path}")
        except Exception as e:
            error_files.append((str(file_path), str(e)))
            print(f"Error deleting {file_path}: {e}")

    # Generate report
    with open('cleanup_report.txt', 'w') as f:
        f.write(f"Cleanup Report - {datetime.datetime.now()}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Age threshold: {args.days} days (before {cutoff_date})\n")
        f.write(f"- Dry run: {args.dry_run}\n")
        f.write(f"- Backup: {args.backup}\n")
        if args.backup:
            f.write(f"- Backup directory: {backup_dir}\n")
        f.write("\n")
        
        f.write(f"Total files deleted: {len(deleted_files)}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Deleted files:\n")
        for file in deleted_files:
            f.write(f"- {file}\n")
        
        if error_files:
            f.write("\nErrors encountered:\n")
            for file, error in error_files:
                f.write(f"- {file}: {error}\n")
    
    print(f"\nCleanup complete. {len(deleted_files)} files deleted.")
    if error_files:
        print(f"{len(error_files)} errors encountered.")
    print(f"See cleanup_report.txt for details.")


if __name__ == "__main__":
    main()
