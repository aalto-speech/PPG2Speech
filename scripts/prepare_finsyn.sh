#!/usr/bin/bash

# Function to generate wav.scp for a folder
generate_wav_scp() {
    local folder="$1"
    local output_file="$2"

    # Ensure the folder exists
    if [ ! -d "$folder" ]; then
        echo "Error: Folder '$folder' does not exist."
        exit 1
    fi

    # Find all .wav files, filter out files containing 'style_' or 'dialogue'
    find "$folder" -type f -name "*.wav" ! -name "*style_*" ! -name "*dialogue*" | while read -r file; do
        # Use the basename without extension as the key and the absolute path as the value
        key=$(basename "$file" .wav)
        echo "$key $(realpath "$file")"
    done >> "$output_file"
}

# Function to split wav.scp into train, val, and test sets
split_wav_scp() {
    local wav_scp="$1"
    local output_dir="$2"
    local train_ratio="$3"
    local val_ratio="$4"

    # Read total number of lines in wav.scp
    total_lines=$(wc -l < "$wav_scp")

    # Calculate number of lines for each split
    train_count=$(printf "%.0f" "$(echo "$total_lines * $train_ratio" | bc)")
    val_count=$(printf "%.0f" "$(echo "$total_lines * $val_ratio" | bc)")
    test_count=$((total_lines - train_count - val_count))

    # Shuffle the lines randomly
    shuffled=$(mktemp)
    shuf "$wav_scp" > "$shuffled"

    # Split into train, val, and test
    mkdir -p "$output_dir/train" "$output_dir/val" "$output_dir/test"
    head -n "$train_count" "$shuffled" > "$output_dir/train/wav.scp"
    tail -n +$((train_count + 1)) "$shuffled" | head -n "$val_count" > "$output_dir/val/wav.scp"
    tail -n "$test_count" "$shuffled" > "$output_dir/test/wav.scp"

    # Clean up
    rm "$shuffled"

    echo "Split completed:"
    echo "Train: $train_count files"
    echo "Validation: $val_count files"
    echo "Test: $test_count files"
}

# Main script logic
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <folder1> <folder2> <output-dir> <wav-scp-filename>"
    exit 1
fi

folder1="$1"
folder2="$2"
output_dir="$3"
wav_scp="$4"

# Temporary combined wav.scp
combined_wav_scp=$(mktemp)

# Generate wav.scp for both folders
generate_wav_scp "$folder1" "$combined_wav_scp"
generate_wav_scp "$folder2" "$combined_wav_scp"

# Remove duplicates from the combined wav.scp
sort -u "$combined_wav_scp" > "$wav_scp"
rm "$combined_wav_scp"

echo "Combined wav.scp created: $wav_scp"

# Split into train, val, and test
split_wav_scp "$wav_scp" "$output_dir" 0.9 0.05