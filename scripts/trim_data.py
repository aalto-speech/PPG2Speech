import os
import sys
import librosa
import soundfile as sf
import torch
from collections import OrderedDict
from pyannote.audio import Pipeline
from loguru import logger
wav_scp_dir = sys.argv[1]
out_dir = sys.argv[2]
token = sys.argv[3]

split = wav_scp_dir.split('/')[-1]

if not os.path.exists(f"{out_dir}/{split}"):
    os.makedirs(f"{out_dir}/{split}/wav")

with open(f"{wav_scp_dir}/wav.scp", "r") as reader:
    lines = reader.readlines()

lines = [line.strip(" \n") for line in lines]

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=token)

write_scp = OrderedDict()

for line in lines:
    key, path = line.split()

    logger.info(f"Processing {path}")

    wav, sr = librosa.load(path=path,
                           sr=None)
    
    duration = librosa.get_duration(y=wav, sr=sr)

    wav_t = torch.from_numpy(wav).unsqueeze(0)
    
    output = pipeline({'waveform': wav_t, 'sample_rate': sr})

    try:
        start = output.get_timeline().support()[0].start
        end = output.get_timeline().support()[-1].end
    except:
        logger.info(f"No significant voice in {key}, move the whole audio.")
        sf.write(f"{out_dir}/{split}/wav/{key}.wav", wav, samplerate=sr)
        write_scp[key] = os.path.abspath(f"{out_dir}/{split}/wav/{key}.wav")
        continue

    start = (start - 0.05) if start > 0.05 else start
    end = (end + 0.05) if end + 0.05 <= duration else duration

    start_frame = librosa.time_to_samples(start, sr=sr)
    end_frame = librosa.time_to_samples(end, sr=sr)

    wav_trimmed = wav[start_frame:end_frame+1]

    logger.info(f"{key}: duration {duration}, voiced start: {start}/{start_frame}, voiced end: {end}/{end_frame}")

    sf.write(f"{out_dir}/{split}/wav/{key}.wav", wav_trimmed, samplerate=sr)

    write_scp[key] = os.path.abspath(f"{out_dir}/{split}/wav/{key}.wav")

with open(f"{out_dir}/{split}/wav.scp", "w") as writer:
    for key, path in write_scp.items():
        writer.write(f"{key} {path}\n")
