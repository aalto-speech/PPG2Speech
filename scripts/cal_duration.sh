#!/usr/bin/bash

wav_scp=$1

module load sox ffmpeg

# Check if wav.scp exists
if [ ! -f "$wav_scp" ]; then
  echo "$wav_scp file not found!"
  exit 1
fi

total_duration=0

while read -r line; do
  # Extract the path to the wav file (second column)
  wav_path=$(echo "$line" | awk '{print $2}')

  # Get duration using soxi
  if [ -f "$wav_path" ]; then
    duration=$(soxi -D "$wav_path")
    total_duration=$(echo "$total_duration + $duration" | bc)
  else
    echo "File not found: $wav_path"
  fi
done < $wav_scp

total_hours=$(echo "scale=2; $total_duration / 3600" | bc)

echo "Total duration: $total_duration seconds"
echo "Total duration: $total_hours hours"