#!/usr/bin/bash

testset=$1
model_class=$2
ckpt=$3
device=$4
vocoder=$5

if [ $# -lt 5 ]; then
    echo "Usage: $0 <testset> <model_class> <ckpt> <device> <vocoder> [<start> <end>]"
    exit 1
fi

start="${6:-0}"
end="${7:-5}"

exp_dir=$(realpath $(dirname "$(dirname "$ckpt")"))

if [ $start -le 0 ] && [ $end -ge 0 ]; then
    echo "Generating mels with the same speaker identity"

    python -m ppg_tts.evaluation.synthesis --model_class ${model_class} \
        --ckpt ${ckpt} --device ${device} --data_dir ${testset}

fi

if [ $start -le 1 ] && [ $end -ge 1 ]; then
    echo "Generating wavs for generated mels"

    if [[ $vocoder == "bigvgan" ]]; then

        curr_dir=$(pwd)

        cd vocoder/bigvgan
        python inference_e2e.py --checkpoint_file bigvgan_generator.pt \
            --input_mels_dir ${exp_dir}/mel --output_dir ${exp_dir[$SLURM_ARRAY_TASK_ID]}/wav_$vocoder

        cd $curr_dir
    else
        python -m vocoder.hifigan.inference_e2e --checkpoint_file vocoder/hifigan/ckpt/g_02500000 \
            --input_mels_dir ${exp_dir}/mel --output_dir ${exp_dir}/wav_$vocoder
    fi
fi

if [ $start -le 2 ] && [ $end -ge 2 ]; then
    echo "Evaluate WER & CER on the synthesized speech"
    python -m ppg_tts.evaluation.evaluate_wer --data_dir ${testset} --flip_wav_dir ${exp_dir}/wav_$vocoder
fi

if [ $start -le 3 ] && [ $end -ge 3 ]; then
    echo "Evaluate MOS score on the synthesized speech"
    python -m ppg_tts.evaluation.mos_eval --flip_wav_dir ${exp_dir}/wav_$vocoder
fi

if [ $start -le 4 ] && [ $end -ge 4 ]; then
    echo "Evaluate pitch dtw and mcd on the synthesized speech"
    python -m ppg_tts.evaluation.evaluate_pitch_mcd --data_dir ${testset} --flip_wav_dir ${exp_dir}/wav_$vocoder
fi