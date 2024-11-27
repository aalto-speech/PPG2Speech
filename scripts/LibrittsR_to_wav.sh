#!/usr/bin/bash

# Define the dataset directory
base_dir=$1 # LibriTTS-R location
dataset_dir=$2 # train-clean-100, etc

# Define the output file

# Clear the output file if it exists
> $base_dir/$dataset_dir/wav.scp

# Traverse the dataset directory and process .wav files
find "$base_dir/$dataset_dir" -type f -name "*.wav" | while read -r filepath; do
    # Extract the recording ID (basename without extension)
    recording_id=$(basename "$filepath" .wav)
    # Write the recording ID and file path to wav.scp
    echo "$recording_id $filepath" >> $base_dir/$dataset_dir/wav.scp
done

echo "wav.scp has been generated and saved to $base_dir/$dataset_dir/wav.scp."
