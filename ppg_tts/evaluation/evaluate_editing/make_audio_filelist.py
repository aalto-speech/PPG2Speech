import sys
from typing import Dict

def build_key_value_mapping(file: str, is_text: bool = False) -> Dict:
    with open(file, 'r') as reader:
        lines = reader.readlines()

    key2value = {}
    if is_text:
        for line in lines:
            key, *text_lst = line.strip(' \n').split()
            key2value[key] = ' '.join(text_lst)
    else:
        for line in lines:
            key, wav_path = line.strip(' \n').split()
            key2value[key] = wav_path

    return key2value

if __name__ == '__main__':
    wav_scp = sys.argv[1]

    edited_text = sys.argv[2]

    output_file = sys.argv[3]

    key2text = build_key_value_mapping(edited_text, True)

    key2wav = build_key_value_mapping(wav_scp)

    writer = open(output_file, 'w', encoding='utf-8')

    for key, wav_path in key2wav.items():
        text = key2text[key]

        spk = key.split('_')[0]

        writer.write(f"{wav_path}|{spk}|{text}\n")

    writer.close()
