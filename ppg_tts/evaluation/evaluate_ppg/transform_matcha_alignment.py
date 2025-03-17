import json
import math
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--edit_json',
        type=str,
    )

    parser.add_argument(
        '--matcha_alignment_folder',
        type=str
    )

    parser.add_argument(
        '--output_json',
        type=str,
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    with open(args.edit_json, 'r') as reader:
        edit_info = json.load(reader)

    matcha_edits_json = {}

    for i, key in enumerate(edit_info):
        matcha_align_json = Path(args.matcha_alignment_folder) / f"{key}.json"

        with open(matcha_align_json.as_posix(), 'r') as reader:
            matcha_align = json.load(reader)

        edit_idx = edit_info[key]['pos_in_str']

        align_info_at_pos = matcha_align[edit_idx]

        # print(f"{key}: {align_info_at_pos}")

        c = list(align_info_at_pos.keys())[0]

        edit_info[key]["edit_region"] = (
            math.ceil(align_info_at_pos[c]['starttime'] * 1.161), #! 22050HZ with 256 hopsize to kaldi 10ms mel frame
            math.ceil(align_info_at_pos[c]['endtime'] * 1.161)
        )

        matcha_edits_json[key] = edit_info[key]

    with open(args.output_json, 'w', encoding='utf-8') as writer:
        json.dump(
            matcha_edits_json,
            writer,
            indent=4,
            ensure_ascii=False,
        )
        
