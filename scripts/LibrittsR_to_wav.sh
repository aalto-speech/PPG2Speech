#!/usr/bin/bash

# Define the dataset directory
dataset_dir=$1 # train-clean-100, etc

# Define the output file

# Clear the output file if it exists
> $dataset_dir/wav.scp

# Traverse the dataset directory and process .wav files
find "$dataset_dir" -type f -name "*.wav" ! -name ".*" | uniq | while read -r filepath; do
    # Extract the recording ID (basename without extension)
    recording_id=$(basename "$filepath" .wav)
    # Write the recording ID and file path to wav.scp
    echo "$recording_id $filepath" >> $dataset_dir/wav.scp
done

echo "wav.scp has been generated and saved to $dataset_dir/wav.scp."
