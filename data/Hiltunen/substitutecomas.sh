#!/bin/bash

# Check if a directory is provided as an argument, otherwise use the current directory
directory="${1:-.}"

# Loop through all .csv files in the given directory
for file in "$directory"/*.csv; do
    # Check if the file exists and is a regular file
    if [[ -f "$file" ]]; then
        # Use sed to replace semicolons with commas
        sed -i 's/;/,/g' "$file"
        echo "Processed file: $file"
    fi
done

echo "All CSV files have been processed."
