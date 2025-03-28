#!/usr/bin/bash

testset=$1
model_class=$2
ckpt=$3
device=$4
vocoder=$5

if [ $# -lt 5 ]; then
    echo "Usage: $0 <testset> <model_class> <ckpt> <device> <vocoder> [<guidance> <sway> <start> <end>]"
    exit 1
fi

guidance="${6:-1.0}"
sway="${7:--1.0}"
start="${8:-0}"
end="${9:-6}"

exp_dir=$(realpath $(dirname "$(dirname "$ckpt")"))
test_dir=$(basename ${testset})

set -e
set -o pipefail

if [ $start -le 0 ] && [ $end -ge 0 ]; then
    echo "Generating mels with flipped speaker identity"

    python -m ppg_tts.evaluation.synthesis --model_class ${model_class} \
        --ckpt ${ckpt} --device ${device} --data_dir ${testset} --switch_speaker \
        --guidance ${guidance} --sway_coeff ${sway}
fi

if [ $start -le 1 ] && [ $end -ge 1 ]; then
    echo "Generating wavs for flipped generated mels"

    if [[ $vocoder == "bigvgan" ]]; then

        curr_dir=$(pwd)

        cd vocoder/bigvgan
        python inference_e2e.py --checkpoint_file bigvgan_generator.pt \
            --input_mels_dir "${exp_dir}/flip_generate_mel_${test_dir}_gd${guidance}_sw${sway}" \
            --output_dir "${exp_dir[$SLURM_ARRAY_TASK_ID]}/flip_generate_wav_${test_dir}_${vocoder}_gd${guidance}_sw${sway}"

        cd $curr_dir
    else
        python -m vocoder.hifigan.inference_e2e \
            --checkpoint_file vocoder/hifigan/ckpt/g_02500000 \
            --input_mels_dir "${exp_dir}/flip_generate_mel_${test_dir}_gd${guidance}_sw${sway}" \
            --output_dir "${exp_dir}/flip_generate_wav_${test_dir}_${vocoder}_gd${guidance}_sw${sway}"
    fi
    cp "${exp_dir}/flip_generate_mel_${test_dir}_gd${guidance}_sw${sway}/speaker_mapping" \
        "${exp_dir}/flip_generate_wav_${test_dir}_${vocoder}_gd${guidance}_sw${sway}/speaker_mapping"
fi

if [ $start -le 2 ] && [ $end -ge 2 ]; then
    echo "Calculate speaker embedding distance between original target speaker wavs and generated target speaker wavs"

    python -m ppg_tts.evaluation.evaluate_spk_emb --data_dir ${testset} \
        --flip_wav_dir "${exp_dir}/flip_generate_wav_${test_dir}_${vocoder}_gd${guidance}_sw${sway}" \
        --device ${device}
fi

if [ $start -le 3 ] && [ $end -ge 3 ]; then
    echo "Evaluate WER & CER on the synthesized speech"
    python -m ppg_tts.evaluation.evaluate_wer --data_dir ${testset} \
        --flip_wav_dir "${exp_dir}/flip_generate_wav_${test_dir}_${vocoder}_gd${guidance}_sw${sway}"
fi

if [ $start -le 4 ] && [ $end -ge 4 ]; then
    echo "Evaluate MOS score on the synthesized speech"
    python -m ppg_tts.evaluation.mos_eval \
        --flip_wav_dir "${exp_dir}/flip_generate_wav_${test_dir}_${vocoder}_gd${guidance}_sw${sway}"
fi
