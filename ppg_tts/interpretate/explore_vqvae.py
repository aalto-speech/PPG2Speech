import torch
import json
from collections import defaultdict, Counter
from einops import rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from loguru import logger
from ..models.AutoEnc.vqvae import VQVAE
from ..evaluation.evaluate_wer import read_wav_scp_text

def load_vqvae(model_path: str) -> VQVAE:
    weights = torch.load(model_path, map_location='cpu')['state_dict']
    vqvae_weights = {k.replace('model.vqvae.', ''): v for k, v in weights.items() if 'vqvae' in k}

    vqvae = VQVAE(
        input_channel=1024,
        hidden_channel=128,
        cond_channel=128,
        kernel_sizes=[3,3,1],
        dilations=[2,4,8],
        num_emb=32,
    )

    ret = vqvae.load_state_dict(vqvae_weights)

    logger.warning(f"Incompatible keys {ret}")

    return vqvae

def get_vqvae_cluster(model: VQVAE, hidden: torch.Tensor) -> torch.Tensor:
    b, t, _ = hidden.shape
    z = rearrange(hidden, 'b t e -> b e t')

    for layer in model.enc:
        z = layer(z)
    z = rearrange(z, 'b e t -> (b t) 1 e')

    centers = model.vq.embedding.weight

    centers = rearrange(centers, "n e -> 1 n e")

    dist_mat = torch.sqrt((z - centers) ** 2).sum(-1) # (b t) n

    center_idx = torch.argmin(dist_mat, dim = -1)

    return rearrange(center_idx, '(b t) -> b t', b=b, t=t)


if __name__ == '__main__':
    w2v2_name = "GetmanY1/wav2vec2-large-fi-150k-finetuned"

    ckpt = "exp3_mixfin_cfm_vqvae_wespeaker_conformer_normpitch_penn/ckpt/epoch=323-step=399492.ckpt"

    logger.info(f"Load Wav2Vec2 model {w2v2_name}")

    processor = Wav2Vec2Processor.from_pretrained(w2v2_name)

    model = Wav2Vec2ForCTC.from_pretrained(w2v2_name)

    logger.info(f"Load VQVAE from ckpt {ckpt}")

    vqvae = load_vqvae(ckpt)

    tkid2clusterid = defaultdict(Counter)

    with torch.no_grad():
        for utterance in read_wav_scp_text('/scratch/elec/t412-speechsynth/DATA/fin-mix/test/wav.scp', '/scratch/elec/t412-speechsynth/DATA/fin-mix/test/text'):
            logger.info(f"Processing file {utterance[-1]}")
            wav = utterance[0]
            text = utterance[1]

            out = model(wav, output_hidden_states=True)

            hidden = out.hidden_states[-1]
            logits = out.logits

            pred_ids = torch.argmax(logits, dim=-1)

            cluster_ids = vqvae.forward(
                hidden,
                cond=torch.randn((1, 128)),
                mask=torch.full((logits.shape[0], logits.shape[1]), False)
            )[-1]

            non_blank_idx = torch.nonzero(pred_ids)

            for idx in non_blank_idx:
                token_id = pred_ids[idx[0], idx[1]].item()

                print(token_id)
                cluster_id = cluster_ids[idx[1]].item()

                match token_id:
                    case 1 | 2 | 3 | 4:
                        continue
                    case _:
                        token = processor.decode(
                            token_ids=token_id
                        )

                        logger.info(f"{utterance[-1]}: {token} bind to cluster {cluster_id}")

                        tkid2clusterid[token][cluster_id] += 1
    
    with open('ppg_tts/explore/cluster.json', 'w') as writer:
        json.dump(tkid2clusterid, writer, indent=4)
