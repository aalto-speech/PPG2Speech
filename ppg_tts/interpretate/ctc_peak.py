import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

if __name__ == '__main__':
    name1 = "GetmanY1/wav2vec2-large-fi-150k-finetuned"
    name2 = "Usin2705/CaptainA_v0"

    processor1 = Wav2Vec2Processor.from_pretrained(name1)
    processor2 = Wav2Vec2Processor.from_pretrained(name2)

    model1 = Wav2Vec2ForCTC.from_pretrained(name1)
    model2 = Wav2Vec2ForCTC.from_pretrained(name2)

    audio_lst = [
        "/scratch/elec/t405-puhe/c/perso_synteesi/female/08/wav/08_test_0004.wav",
        "/scratch/elec/t405-puhe/c/perso_synteesi/male/11m/wav/11m_test_0005.wav",
        "/scratch/elec/t405-puhe/c/perso_synteesi/female/17/wav/17_test_0005.wav",
        "/scratch/elec/t405-puhe/c/perso_synteesi/male/14m/wav/14m_test_0009.wav",
        "/scratch/elec/t405-puhe/c/perso_synteesi/female/11/wav/11_test_0001.wav"
    ]

    writer = open('ctc_peak.txt', 'w')

    formater = lambda name, out1, out2, pred1, pred2: f"{name}\nctc_out: {out1}\nctc_ce_out: {out2}\nctc_pred: {pred1}\nctc_ce_pred: {pred2}"

    for audio in audio_lst:
        x, sr = torchaudio.load(audio)

        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)

        with torch.no_grad():
            output1 = torch.argmax(model1(x).logits, dim=-1)
            output2 = torch.argmax(model2(x).logits, dim=-1)

            label_idx_1 = output1.squeeze(0).tolist()
            label_idx_2 = output2.squeeze(0).tolist()

            pred1 = processor1.batch_decode(
                output1
            )[0].lower()
            pred2 = processor2.batch_decode(
                output2
            )[0].lower()

        print(formater(audio, label_idx_1, label_idx_2, pred1, pred2), file=writer, flush=True)
