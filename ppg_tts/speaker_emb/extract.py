from ..utils import build_parser
from ..dataset import PersoDataset
from .EmbeddingModels import SpeakerEmbeddingPretrained
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDataset(args.data_dir)
    SpEmModel = SpeakerEmbeddingPretrained(args.auth_token)

    with WriteHelper(f"ark,scp:{args.data_dir}/embedding.ark,{args.data_dir}/embedding.scp") as writer:
        for i, (key, utterance) in enumerate(dataset):
            wav = utterance["feature"]
            emb = SpEmModel.forward(wav)
            writer(str(key), emb.numpy())
