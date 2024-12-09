#!/bin/bash

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <destination> <source1> [<source2> ...]"
    exit 1
fi

# Extract destination path
destination="$1"
shift  # Shift arguments to process source datasets

# Define subsets and special mapping for LibriTTS
declare -A libri_mapping=(
    ["train"]="train-clean-100"
    ["val"]="dev-clean"
    ["test"]="test-clean"
)
subsets=("train" "val" "test")
files=("wav.scp" "log_f0.scp" "embedding.scp" "voiced.scp" "ppg_no_ctc.scp")

# Create destination subsets if not exist
for subset in "${subsets[@]}"; do
    dest_subset="$destination/$subset"
    mkdir -p "$dest_subset"
    for file in "${files[@]}"; do
        > "$dest_subset/$file"  # Clear/create files
    done
done

# Process each source dataset
for source in "$@"; do
    if [[ "$source" == *"LibriTTS"* ]]; then
        # Handle LibriTTS subset mapping
        for original_subset in "${!libri_mapping[@]}"; do
            mapped_subset="${libri_mapping[$original_subset]}"
            source_subset="$source/$mapped_subset"
            dest_subset="$destination/$original_subset"

            if [ -d "$source_subset" ]; then
                for file in "${files[@]}"; do
                    src_file="$source_subset/$file"
                    dest_file="$dest_subset/$file"
                    if [ -f "$src_file" ]; then
                        echo "Merging $src_file into $dest_file"
                        cat "$src_file" >> "$dest_file"
                    else
                        echo "Warning: File $src_file does not exist."
                    fi
                done
            else
                echo "Warning: Directory $source_subset does not exist."
            fi
        done
    else
        # Handle normal subset names
        for subset in "${subsets[@]}"; do
            source_subset="$source/$subset"
            dest_subset="$destination/$subset"

            if [ -d "$source_subset" ]; then
                for file in "${files[@]}"; do
                    src_file="$source_subset/$file"
                    dest_file="$dest_subset/$file"
                    if [ -f "$src_file" ]; then
                        echo "Merging $src_file into $dest_file"
                        cat "$src_file" >> "$dest_file"
                    else
                        echo "Warning: File $src_file does not exist."
                    fi
                done
            else
                echo "Warning: Directory $source_subset does not exist."
            fi
        done
    fi
done

echo "Merging completed."
