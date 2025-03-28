#!/usr/bin/bash

# Check if the correct number of arguments is given
if [ $# -ne 2 ]; then
    echo "Usage: $0 /path/to/wav/folder /path/to/text/file"
    exit 1
fi

WAV_DIR=$1
TEXT_FILE=$2
OUTPUT_DIR=$WAV_DIR/kaldi_dataset

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define output files
WAV_SCP="$OUTPUT_DIR/wav.scp"
UTT2SPK="$OUTPUT_DIR/utt2spk"

# Clear existing files
> "$WAV_SCP"
> "$UTT2SPK"

# Process each wav file
for file in "$WAV_DIR"/*.wav; do
    [ -e "$file" ] || continue  # Skip if no wav files exist

    # Get the absolute path
    abs_path=$(realpath "$file")

    # Extract key from filename (e.g., 19_test_0008 from 19_test_0008_generated_e2e.wav)
    filename=$(basename "$file")
    key=$(echo "$filename" | sed -E 's/_generated_e2e\.wav//')

    # Write to wav.scp (use SoX for resampling)
    echo "$key sox $abs_path -r 16k -b 16 -t wav - |" >> "$WAV_SCP"

    # Write to utt2spk
    speaker=$(echo "$key" | cut -d'_' -f1)
    echo "$key $speaker" >> "$UTT2SPK"
done

# Copy the text file
# cp -L "$TEXT_FILE" "$OUTPUT_DIR/text" || { echo "Error copying text file"; exit 1; }
awk '{ for(i=2; i<=NF; i++) { $i=tolower($i); gsub(/[^a-zäöå0-9]/, "", $i) } print }' "$TEXT_FILE" > "$OUTPUT_DIR/text"

echo "Kaldi dataset files created in $OUTPUT_DIR:"
echo " - wav.scp (resampled to 16kHz using SoX)"
echo " - utt2spk"
echo " - text (copied and cleaned from $TEXT_FILE)"
