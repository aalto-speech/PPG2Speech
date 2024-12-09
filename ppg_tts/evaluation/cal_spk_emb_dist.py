import torch
import torchaudio
import shutil
from loguru import logger
from torch.nn import CosineSimilarity
from ..dataset import ExtendDataset
from ..models import SpeakerEmbeddingPretrained
from ..utils import build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = ExtendDataset(data_dir=args.data_dir)

    SpEmModel = SpeakerEmbeddingPretrained(args.auth_token, args.device)

    with open(f"{args.flip_wav_dir}/speaker_mapping", "r") as reader:
        mapping = reader.readlines()
    
    cos = CosineSimilarity(dim=0)
    avg_simi = 0.0
    for entry in mapping:
        source_key, target_key = entry.strip("\n").split()

        source_wav_path = f"{args.flip_wav_dir}/{source_key}_generated_e2e.wav"
        target_idx = dataset.key2idx[target_key]

        source_wav, source_sr = torchaudio.load(source_wav_path)

        source_wav = torchaudio.functional.resample(source_wav, source_sr, 16000)

        source_spk_emb = torch.from_numpy(SpEmModel.forward(source_wav, 16000))
        target_spk_emb = dataset[target_idx]['spk_emb']

        with torch.no_grad():
            similarity = cos(source_spk_emb, target_spk_emb)

        avg_simi += similarity.detach().cpu().item()

        logger.info(f"Cosine similarity between {source_key} and {target_key} is {similarity.detach().cpu().item()}")

        target_wav_path = dataset.key2wav[target_key]

        shutil.copyfile(target_wav_path, f"{args.flip_wav_dir}/{source_key}_style_reference.wav")

    avg_simi /= len(dataset)

    logger.info(f"The average cosine similarity is {avg_simi}")
