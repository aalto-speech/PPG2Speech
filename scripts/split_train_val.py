import sys
import soundfile as sf
from typing import Dict, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

def write(key_value: Dict, filename: Path):
    print(f"Write to {filename}")
    with open(filename, "w") as writer:
        for k, v in key_value.items():
            writer.write(f"{k} {v}")

def get_audio_length(audio_file: str) -> float:
    audio_file = audio_file.strip("\n")
    with sf.SoundFile(audio_file) as af:
        duration = len(af) / af.samplerate
    return duration

def sort_wav_text_on_key(wavs: Dict, texts: Dict) -> Tuple[Dict, Dict]:
    sorted_kvs = sorted(wavs.items(),
                        key=lambda kv: get_audio_length(kv[1]))
    sorted_wav = dict(sorted_kvs)

    sorted_texts = {k: texts[k] for k, _ in sorted_kvs}

    return sorted_wav, sorted_texts

if __name__ == "__main__":
    train_val_dir = Path(sys.argv[1])
    test_size = float(sys.argv[2])

    with open(train_val_dir / "text", "r") as text_reader:
        print("Read texts file")
        texts = text_reader.readlines()

    with open(train_val_dir / "wav.scp", "r") as scp_reader:
        print("Read wav.scp file")
        wavs = scp_reader.readlines()

    texts_dict = {key: " ".join(words) 
                  for key, *words in map(lambda x: x.split(" "), texts)}
    wavs_dict = {key: wav for key, wav in map(lambda x: x.split(" "), wavs)}

    keys = list(texts_dict.keys())

    print(f"Split into train and val set, val set ratio {test_size}")
    keys_train, keys_val = train_test_split(keys,
                                            test_size=test_size,
                                            random_state=17)
    
    pair_func = lambda keys, dict: {key: dict[key] for key in keys}
    train_texts = pair_func(keys_train, texts_dict)
    train_wavs = pair_func(keys_train, wavs_dict)

    val_texts = pair_func(keys_val, texts_dict)
    val_wavs = pair_func(keys_val, wavs_dict)

    print("Sort the train and val set based on audio duration.")
    train_wavs, train_texts = sort_wav_text_on_key(train_wavs, train_texts)
    val_wavs, val_texts = sort_wav_text_on_key(val_wavs, val_texts)

    parent_path = train_val_dir.parent

    print("Write the dataset to disk.")
    write(train_texts, parent_path / "train/text")
    write(train_wavs, parent_path / "train/wav.scp")

    write(val_texts, parent_path / "val/text")
    write(val_wavs, parent_path / "val/wav.scp")
