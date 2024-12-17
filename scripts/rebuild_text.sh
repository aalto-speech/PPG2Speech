#!/bin/bash

# Usage: ./script.sh <base_path> <text_file>

# Check if sufficient arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <base_path> <text_file>"
  exit 1
fi

# Variables
BASE_PATH=$1
TEXT_FILE=$2
BACKUP_FILE="${TEXT_FILE}.old"
NEW_TEXT_FILE=${TEXT_FILE}
ERROR_KEYS_FILE="$(dirname "$TEXT_FILE")/error_keys.txt"
WAV_SCP_FILE="$(dirname "$TEXT_FILE")/wav.scp"

# Step 1: Backup the original text file
mv "$TEXT_FILE" "$BACKUP_FILE"

# Initialize the error keys file
> "$ERROR_KEYS_FILE"

# Step 2: Process the backup file
> "$NEW_TEXT_FILE" # Create or clear the new text file

while IFS=' ' read -r key content; do
  # Extract speaker ID from the key (first column)
  speaker_id=$(echo "$key" | cut -d'_' -f1)

  # Determine gender path based on speaker ID
  if [[ "$speaker_id" == *m* ]]; then
    prompt_file="${BASE_PATH}/male/${speaker_id}/prompts/${key}.prompt"
  else
    prompt_file="${BASE_PATH}/female/${speaker_id}/prompts/${key}.prompt"
  fi

  # Check if the prompt file exists
  if [[ -f "$prompt_file" ]]; then
    # Detect the encoding of the prompt file
    ENCODING=$(file -bi "$prompt_file" | sed -n 's/.*charset=//p')

    # If the encoding is not UTF-8, try to convert it
    if [[ "$ENCODING" != "utf-8" ]]; then
      converted_prompt=$(iconv -f "$ENCODING" -t utf-8 "$prompt_file") || {
        echo "Error: Unable to convert prompt file $prompt_file to UTF-8." >&2
        echo "$key" >> "$ERROR_KEYS_FILE"  # Save the key of the problematic file
        continue
      }
    else
      # If already UTF-8, just read the content
      converted_prompt=$(<"$prompt_file")
    fi

    # Append the key and converted prompt content to the new text file
    echo -e "$key $converted_prompt" >> "$NEW_TEXT_FILE"
  else
    echo "Warning: Prompt file not found for key: $key" >&2
  fi
done < "$BACKUP_FILE"

# Step 3: Filter the wav.scp file to remove lines with error keys
if [[ -f "$WAV_SCP_FILE" ]]; then
  # Create a backup of the wav.scp before modifying
  cp "$WAV_SCP_FILE" "${WAV_SCP_FILE}.bak"
  
  # Filter out lines containing keys that couldn't be converted
  grep -v -f "$ERROR_KEYS_FILE" "$WAV_SCP_FILE" > "${WAV_SCP_FILE}.filtered"

  # Replace original wav.scp with the filtered version
  mv "${WAV_SCP_FILE}.filtered" "$WAV_SCP_FILE"
else
  echo "Warning: wav.scp file not found in the same folder as $TEXT_FILE." >&2
fi

# Print completion message
echo "Processing complete. New text file created: $NEW_TEXT_FILE"
echo "Keys that couldn't be converted have been saved in: $ERROR_KEYS_FILE"
