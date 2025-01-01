#!/usr/bin/bash

testset=$1
ckpt=$2
device=$3
vocoder=$4

if [ $# -lt 4 ]; then
    echo "Usage: $0 <testset> <ckpt> <device> <vocoder>"
    exit 1
fi

exp_dir=$(realpath $(dirname "$(dirname "$ckpt")"))

echo "Generating mels with flipped speaker identity"

python -m ppg_tts.evaluation.synthesis --ckpt ${ckpt} --device ${device} --data_dir ${testset}

echo "Generating wavs for flipped generated mels"

if [[ $vocoder == "bigvgan" ]]; then

    curr_dir=$(pwd)

    cd vocoder/bigvgan
    python inference_e2e.py --checkpoint_file bigvgan_generator.pt --input_mels_dir ${exp_dir}/mel --output_dir ${exp_dir[$SLURM_ARRAY_TASK_ID]}/wav_$vocoder

    cd $curr_dir
else
    python -m vocoder.hifigan.inference_e2e --checkpoint_file vocoder/hifigan/ckpt/g_02500000 --input_mels_dir ${exp_dir}/mel --output_dir ${exp_dir}/wav_$vocoder
fi

echo "Evaluate WER & CER on the synthesized speech"
python -m ppg_tts.evaluation.evaluate_wer --data_dir ${testset} --flip_wav_dir ${exp_dir}/wav_$vocoder

echo "Evaluate MOS score on the synthesized speech"
python -m ppg_tts.evaluation.dnsmos_eval --flip_wav_dir ${exp_dir}/wav_$vocoder
