from ..utils import build_parser
from ..dataset import PersoDatasetBasic
from ..models import SpeakerEmbeddingPretrained
from kaldiio import WriteHelper
from tqdm import tqdm

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)
    SpEmModel = SpeakerEmbeddingPretrained(args.auth_token, args.device)

    print(f"Extracting Speaker Embedding to {args.data_dir}")

    with WriteHelper(f"ark,scp:{args.data_dir}/embedding.ark,{args.data_dir}/embedding.scp") as writer:
        for i, utterance in tqdm(enumerate(dataset)):
            wav = utterance["feature"]
            emb = SpEmModel.forward(wav)
            writer(utterance['key'], emb)
