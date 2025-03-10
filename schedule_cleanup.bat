@echo off
REM This batch file runs the cleanup_old_files.py script 
REM with appropriate options to delete files older than 2 days.

echo Running file cleanup script...
echo Removing files older than 2 days...

REM Run the Python script with backup enabled
python cleanup_old_files.py --backup --no-confirm

REM Display the cleanup report
echo.
echo Cleanup report:
echo --------------
type cleanup_report.txt
echo --------------
echo.

echo Cleanup operation complete.
echo A backup of deleted files was created.
echo.
