BEGIN {
    FS = " ";      # Use space as the field separator for wav.scp
    OFS = " ";     # Set output field separator to space for the final output file
}

FNR == NR {
    # Read keys from wav.scp and store them in a map (wav_keys)
    wav_keys[$1]
    next
}

{
    # Extract the key from the text file by splitting on '|'
    split($0, arr, "|");
    key_with_path = arr[1];  # Full file path with '.wav'
    text = arr[2];           # The associated text (everything after '|')

    # Extract the key (without the path or extension) from the full file path
    split(key_with_path, key_arr, "/");
    key = key_arr[length(key_arr)];  # Last part is the key with '.wav'
    sub(".wav$", "", key);  # Remove the '.wav' extension
    
    # Check if the extracted key exists in wav_keys
    if (key in wav_keys) {
        print key, text  # Output the key and text with a space separator
    }
}
