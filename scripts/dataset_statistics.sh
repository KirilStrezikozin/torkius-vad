#!/usr/bin/env bash
# Copyright (c) 2026 Kiril Strezikozin
#
# SPDX-License-Identifier: MIT
#
# This script computes statistics for audio datasets stored in subdirectories
# of "sets". It calculates total size, number of files, total duration in
# seconds and hours. It outputs the results in CSV format to stdout.

# Get user confirmation.
confirmation=""
read -p "This script will analyze audio files in subdirs of the current dir. Do you want to proceed? (Y/n): " confirmation
if [[ -n "$confirmation" && "$confirmation" != "y" ]]; then
	echo "Cancelled."
	exit 0
fi

# Print CSV header
echo "directory,total_size_mb,total_files,total_seconds,total_hours"

for dir in ./*/; do
    # Remove trailing slash for cleaner output.
    dirname=$(basename "$dir")
    
    # Find audio files.
    files=$(find "$dir" -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" \))
    
    # Skip if no files found.
    if [ -z "$files" ]; then
        echo "$dirname,0,0,0,0"
        continue
    fi
    
    # Total number of files.
    total_files=$(echo "$files" | wc -l)
    
	# Total size in bytes.
	total_bytes=$(du -sb $files | awk '{sum+=$1} END{print sum}')

	# Convert total in bytes to MB.
	total_size_mb=$(awk -v bytes="$total_bytes" 'BEGIN {printf "%f", bytes/(1024*1024)}')
    
	# Total duration in seconds (parallelized).
    total_seconds=$(echo "$files" | tr '\n' '\0' \
        | parallel -0 ffprobe -v error -show_entries format=duration -of csv=p=0 {} \
        | awk '{sum+=$1} END{print sum}')
    
    # Total hours.
    total_hours=$(awk -v sec="$total_seconds" 'BEGIN {printf "%f", sec/3600}')
    
    # Print CSV row.
    echo "$dirname,$total_size_mb,$total_files,$total_seconds,$total_hours"
done
