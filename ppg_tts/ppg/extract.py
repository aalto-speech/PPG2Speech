from ..utils import build_parser
from ..dataset import PersoDataset
from .ASRModels import PPGFromWav2Vec2Pretrained
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDataset(args.data_dir)
    ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained)

    with WriteHelper(f"ark,scp:{args.data_dir}/ppg.ark,{args.data_dir}/ppg.scp") as writer:
        for i, (key, utterance) in enumerate(dataset):
            wav = utterance["feature"]
            ppg = ASRModel.forward(wav)
            writer(str(key), ppg.numpy())