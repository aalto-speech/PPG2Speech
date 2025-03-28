import os
import sys
import glob

def main():
    if len(sys.argv) != 3:
        print("Usage: {} lab_folder output_file".format(sys.argv[0]))
        sys.exit(1)
        
    lab_folder = sys.argv[1]
    output_file = sys.argv[2]
    
    # Dictionary to store word to transcription mapping.
    lexicon = {}

    # Process all .lab files in subdirectories (each subfolder is assumed to be a speaker ID)
    lab_files = glob.glob(os.path.join(lab_folder, '*', '*.lab'))
    for lab_file in lab_files:
        with open(lab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split the transcript line into words by whitespace.
                words = line.split()
                for word in words:
                    # Create transcription: split word into characters separated by space.
                    transcription = " ".join(list(word))
                    # Add the transcription to the lexicon, ensuring no duplicates.
                    if word in lexicon:
                        if transcription not in lexicon[word]:
                            lexicon[word].append(transcription)
                    else:
                        lexicon[word] = [transcription]
    
    # Write the lexicon to the output file.
    with open(output_file, 'w', encoding='utf-8') as out:
        for word, transcriptions in sorted(lexicon.items()):
            for transcription in transcriptions:
                out.write(f"{word}\t{transcription}\n")
    
    print(f"Lexicon file written to {output_file}")

if __name__ == '__main__':
    main()
