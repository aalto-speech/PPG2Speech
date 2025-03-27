import shutil
import wespeaker
from loguru import logger
from ..dataset import ExtendDataset
from ..utils import build_parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = ExtendDataset(data_dir=args.data_dir)
    SpEmModel = wespeaker.load_model('vblinkf')

    unseen_data = ExtendDataset(data_dir='/scratch/elec/t412-speechsynth/DATA/fin-mix/test_unseen')

    logger.add(
        f"{args.flip_wav_dir}/logs/SpkSim.log",
        rotation='200 MB'
    )

    with open(f"{args.flip_wav_dir}/speaker_mapping", "r") as reader:
        mapping = reader.readlines()
    
    avg_simi = 0.0
    avg_unseen_simi = 0.0
    is_unseen = False
    num_unseen = 0
    for i, entry in enumerate(mapping):
        source_key, target_key = entry.strip("\n").split()

        target_speaker = target_key.split('_')[0]

        synthesized_wav_path = f"{args.flip_wav_dir}/{source_key}_generated_e2e.wav"
        try:
            source_wav_path = dataset.key2wav[source_key]
        except:
            source_wav_path = unseen_data.key2wav[source_key]

        try:
            target_wav_path = dataset.key2wav[target_key]
        except:
            is_unseen = True
            num_unseen += 1
            target_wav_path = unseen_data.key2wav[target_key]

        similarity = SpEmModel.compute_similarity(
            synthesized_wav_path,
            target_wav_path
        )

        if is_unseen:
            avg_unseen_simi += (similarity - avg_unseen_simi) / (num_unseen + 1)
            is_unseen = False
        else:
            avg_simi += (similarity - avg_simi) / (i - num_unseen + 1)

        logger.info(
            f"Cosine similarity between {source_key} and {target_key} is {similarity},"
        )

        if args.debug:
            shutil.copyfile(target_wav_path, f"{args.flip_wav_dir}/{source_key}_speaker_reference.wav")

            shutil.copyfile(source_wav_path, f"{args.flip_wav_dir}/{source_key}_context_reference.wav")

    logger.info(f"The average cosine similarity for seen speaker is {avg_simi}, for unseen speaker is {avg_unseen_simi}")
