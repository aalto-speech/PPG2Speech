#!/usr/bin/bash

# Path to the VCTK dataset
VCTK_PATH=$WRKDIR/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed
OUTPUT_PATH=$WRKDIR/vctk_split

# Create output directories
mkdir -p "$OUTPUT_PATH/train"
mkdir -p "$OUTPUT_PATH/val"
mkdir -p "$OUTPUT_PATH/test"

# Initialize wav.scp files
echo -n > "$OUTPUT_PATH/train/wav.scp"
echo -n > "$OUTPUT_PATH/val/wav.scp"
echo -n > "$OUTPUT_PATH/test/wav.scp"

# Iterate over speaker folders
for speaker in "$VCTK_PATH"/*/; do
    if [[ ! -d $speaker ]]; then
        continue
    fi
    speaker_id=$(basename "$speaker")
    
    # Skip p280 and p362
    if [[ "$speaker_id" == "p280" || "$speaker_id" == "p362" ]]; then
        continue
    fi

    echo "Processing speaker ${speaker_id}"
    
    # Find all mic2 audio files for the current speaker
    mic2_files=($(find "$speaker" -type f -name "*mic2*"))
    
    # Shuffle files for random split
    shuffled_files=($(shuf -e "${mic2_files[@]}"))
    total_files=${#shuffled_files[@]}
    
    # Calculate split sizes
    train_size=$((total_files * 80 / 100))
    val_size=$((total_files * 10 / 100))
    test_size=$((total_files - train_size - val_size))
    
    # Split files into train, val, and test
    for i in "${!shuffled_files[@]}"; do
        file=${shuffled_files[$i]}
        file_name=$(basename "$file")
        rel_path="${speaker_id}/$file_name"
        
        # Determine destination split and scp file
        if [ $i -lt $train_size ]; then
            dest="$OUTPUT_PATH/train/$rel_path"
            scp_file="$OUTPUT_PATH/train/wav.scp"
        elif [ $i -lt $((train_size + val_size)) ]; then
            dest="$OUTPUT_PATH/val/$rel_path"
            scp_file="$OUTPUT_PATH/val/wav.scp"
        else
            dest="$OUTPUT_PATH/test/$rel_path"
            scp_file="$OUTPUT_PATH/test/wav.scp"
        fi
        
        # Create directories and move files
        mkdir -p "$(dirname "$dest")"
        cp "$file" "$dest"
        
        # Write to wav.scp
        key="${speaker_id}_${file_name%_*}" # Remove mic2 from key
        echo "$key $dest" >> "$scp_file"
    done
done

echo "Dataset processing completed! wav.scp files are created for each split."
