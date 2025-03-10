# File Cleanup Utility

This utility helps you delete graph and table files older than a specified time period (default: 2 days). It targets PNG, CSV, HTML, TEX, and JSON files in various directories of the Recursive Geometric Quantum Scaling project.

## Files

- `cleanup_old_files.py` - The main Python script that identifies and removes old files
- `schedule_cleanup.bat` - A batch file for easy execution on Windows

## Manual Usage

### Basic Usage

For safety, always run in dry-run mode first to see what would be deleted:

```
python cleanup_old_files.py --dry-run
```

Then, to actually delete files:

```
python cleanup_old_files.py
```

### Options

- `--dry-run` - Show what would be deleted without actually deleting
- `--backup` - Create backups before deletion
- `--days DAYS` - Age threshold in days (default: 2.0)
- `--no-confirm` - Skip confirmation prompt

### Examples

Delete files older than 5 days:
```
python cleanup_old_files.py --days 5
```

Delete files without confirmation but with backup:
```
python cleanup_old_files.py --backup --no-confirm
```

## Scheduling Automatic Cleanup

### Using the Batch File

You can run the included batch file:
```
schedule_cleanup.bat
```

This will run the script with backups enabled and confirmation disabled.

### Setting Up Windows Task Scheduler

To schedule the cleanup to run automatically:

1. Open Windows Task Scheduler
2. Create a new basic task
3. Name it "Quantum Simulation Cleanup"
4. Choose when to run it (e.g., daily)
5. Choose "Start a program"
6. Browse to the batch file location: `C:\Users\17175\Desktop\recursive-geometric-quantum-scaling\schedule_cleanup.bat`
7. Finish the setup

## Directories Scanned

The following directories are scanned for old files:

- `plots/`
- `report/`
- `data/`
- `validation_results/`
- `test_results/` (including various subfolders)

## File Types Targeted

- Images: `.png`, `.jpg`, `.jpeg`, `.gif`
- Tables: `.csv`, `.html`, `.tex`
- Data files: `.json`

## Safety Features

- Dry run mode
- Confirmation prompt by default
- Backup capability
- Report generation
- Error handling

## Generated Reports

After each run, a file named `cleanup_report.txt` is created containing:

- Parameters used
- List of deleted files
- Any errors encountered

## Backup Files

If the `--backup` option is used, a backup directory is created with the naming convention:

```
file_backups_YYYYMMDD_HHMMSS
```

All deleted files are copied to this directory before deletion.
