import os
import argparse

def build_prosodylab_corpus(text_file, audio_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Parse the line
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        unique_id, text = parts

        # Extract speaker ID (part before the first '_')
        speaker_id = unique_id.split('_')[0]

        # Create speaker directory inside the output directory
        speaker_dir = os.path.join(output_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        # Write the text to a .lab file named after the unique ID
        lab_file_path = os.path.join(speaker_dir, f"{unique_id}.lab")
        with open(lab_file_path, 'w', encoding='utf-8') as lab_file:
            lab_file.write(text)

        # Find the corresponding audio file
        audio_file_name = f"{unique_id}_generated_e2e.wav"
        audio_file_path = os.path.join(audio_dir, audio_file_name)

        # Create a soft link to the audio file in the speaker directory
        if os.path.exists(audio_file_path):
            symlink_path = os.path.join(speaker_dir, f"{unique_id}.wav")  # Discard 'generated_e2e' suffix
            if os.path.exists(symlink_path) or os.path.islink(symlink_path):  # Remove existing symlink or file
                os.remove(symlink_path)
            os.symlink(os.path.abspath(audio_file_path), symlink_path)  # Use absolute path for the symlink
        else:
            print(f"Warning: Audio file not found for {unique_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a prosodylab_corpus_directory style folder.")
    parser.add_argument("--text_file", help="Path to the text file")
    parser.add_argument("--audio_dir", help="Path to the directory containing audio files")
    parser.add_argument("--output_dir", help="Path to the output directory")

    args = parser.parse_args()
    build_prosodylab_corpus(args.text_file, args.audio_dir, args.output_dir)