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

    logger.add(
        f"{args.flip_wav_dir}/logs/SpkSim.log",
        rotation='200 MB'
    )

    with open(f"{args.flip_wav_dir}/speaker_mapping", "r") as reader:
        mapping = reader.readlines()
    
    avg_simi = 0.0
    for entry in mapping:
        source_key, target_key = entry.strip("\n").split()

        synthesized_wav_path = f"{args.flip_wav_dir}/{source_key}_generated_e2e.wav"
        source_wav_path = dataset.key2wav[source_key]
        target_wav_path = dataset.key2wav[target_key]
        target_idx = dataset.key2idx[target_key]

        similarity = SpEmModel.compute_similarity(
            synthesized_wav_path,
            target_wav_path
        )

        avg_simi += similarity

        logger.info(
            f"Cosine similarity between {source_key} and {target_key} is {similarity},"
        )

        if args.debug:
            shutil.copyfile(target_wav_path, f"{args.flip_wav_dir}/{source_key}_speaker_reference.wav")

            shutil.copyfile(source_wav_path, f"{args.flip_wav_dir}/{source_key}_context_reference.wav")

    avg_simi /= len(dataset)

    logger.info(f"The average cosine similarity is {avg_simi}")
