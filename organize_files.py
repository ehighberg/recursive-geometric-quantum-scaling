#!/usr/bin/env python
"""
Organize files in the root directory by moving plots and test images to appropriate folders.
"""

import os
import shutil
from pathlib import Path

def organize_files():
    """
    Organize files in the root directory by moving them to appropriate folders.
    """
    print("Organizing files in the root directory...")
    
    # Create directories if they don't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    test_images_dir = Path("test_images")
    test_images_dir.mkdir(exist_ok=True)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # List of files to keep in the root directory
    keep_files = [
        "__init__.py",
        ".gitignore",
        ".pylintrc",
        "app.py",
        "constants.py",
        "fix_init.py",
        "README.md",
        "requirements.txt",
        "TABLE_OF_CONTENTS.md",
        "PAPER_GUIDE.md",
        "run_evolution_analysis.py",
        "generate_summary_table.py",
        "generate_report.py",
        "organize_files.py"
    ]
    
    # List of directories to keep in the root directory
    keep_dirs = [
        "analyses",
        "app",
        "config",
        "docs",
        "simulations",
        "tests",
        "plots",
        "test_images",
        "data",
        "report"
    ]
    
    # Move files to appropriate directories
    for file in os.listdir("."):
        # Skip directories and files to keep
        if os.path.isdir(file) or file in keep_files:
            continue
        
        # Determine destination based on file extension and name
        if file.endswith((".png", ".gif")) and file.startswith("test_"):
            # Test images
            dest_dir = test_images_dir
        elif file.endswith((".png", ".gif")):
            # Plot images
            dest_dir = plots_dir
        elif file.endswith((".csv", ".tex")):
            # Data files
            dest_dir = data_dir
        else:
            # Skip other files
            print(f"Skipping {file}")
            continue
        
        # Move file
        try:
            shutil.move(file, dest_dir / file)
            print(f"Moved {file} to {dest_dir}")
        except Exception as e:
            print(f"Error moving {file}: {e}")
    
    print("File organization complete.")
    
    # Print summary
    print("\nSummary of organized files:")
    print(f"Test images in {test_images_dir}: {len(os.listdir(test_images_dir))}")
    print(f"Plot images in {plots_dir}: {len(os.listdir(plots_dir))}")
    print(f"Data files in {data_dir}: {len(os.listdir(data_dir))}")

if __name__ == "__main__":
    organize_files()
