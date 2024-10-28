import sys
from typing import Dict
from pathlib import Path
from sklearn.model_selection import train_test_split

def write(key_value: Dict, filename: Path):
    print(f"Write to {filename}")
    with open(filename, "w") as writer:
        for k, v in key_value.items():
            writer.write(f"{k} {v}")

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

    keys_train, keys_val = train_test_split(keys,
                                            test_size=test_size,
                                            random_state=17)
    
    pair_func = lambda keys, dict: {key: dict[key] for key in keys}
    train_texts = pair_func(keys_train, texts_dict)
    train_wavs = pair_func(keys_train, wavs_dict)

    val_texts = pair_func(keys_val, texts_dict)
    val_wavs = pair_func(keys_val, wavs_dict)

    parent_path = train_val_dir.parent

    write(train_texts, parent_path / "train/texts")
    write(train_wavs, parent_path / "train/wav.scp")

    write(val_texts, parent_path / "val/texts")
    write(val_wavs, parent_path / "val/wav.scp")
