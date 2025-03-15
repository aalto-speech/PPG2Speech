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
test_dir=$(basename ${testset})


if [ $start -le 0 ] && [ $end -ge 0 ]; then
    echo "Generating mels with flipped speaker identity"

    python -m ppg_tts.evaluation.synthesis --model_class ${model_class} \
        --ckpt ${ckpt} --device ${device} --data_dir ${testset} --edit_ppg
fi

if [ $start -le 1 ] && [ $end -ge 1 ]; then
    echo "Generating wavs for edited mels"

    if [[ $vocoder == "bigvgan" ]]; then

        curr_dir=$(pwd)

        cd vocoder/bigvgan
        python inference_e2e.py --checkpoint_file bigvgan_generator.pt \
            --input_mels_dir ${exp_dir}/editing/mel_${test_dir} \
            --output_dir ${exp_dir[$SLURM_ARRAY_TASK_ID]}/editing/wav_${test_dir}_$vocoder

        cd $curr_dir
    else
        python -m vocoder.hifigan.inference_e2e \
            --checkpoint_file vocoder/hifigan/ckpt/g_02500000 \
            --input_mels_dir ${exp_dir}/editing/mel_${test_dir} \
            --output_dir ${exp_dir}/editing/wav_${test_dir}_$vocoder
    fi
    cp ${exp_dir}/editing/mel_${test_dir}/speaker_mapping \
        ${exp_dir}/editing/wav_${test_dir}_$vocoder/speaker_mapping
fi


if [ $start -le 2 ] && [ $end -ge 2 ]; then
    echo "Inference TTS-baseline for edited text";

    srun python ppg_tts/evaluation/evaluate_ppg/tts_baseline/inference_matcha.py \
        ppg_tts/evaluation/evaluate_ppg/tts_baseline/matcha_hifigan.onnx \
        --file ${exp_dir}/editing/text \
        --spk ${testset}/embedding.scp \
        --output-dir ${exp_dir}/editing/wav_baseline_${test_dir}_$vocoder
fi

if [ $start -le 3 ] && [ $end -ge 3 ]; then
    echo "Extract PPG from synthesized speech"

    echo "....Extracting PPG from ${exp_dir}/editing/wav_baseline_${test_dir}_$vocoder";

    ./ppg_tts/evaluation/evaluate_ppg/extract_kaldi_ppg.sh \
        --wav_dir ${exp_dir}/editing/wav_baseline_${test_dir}_$vocoder \
        --text_file ${exp_dir}/editing/text

    echo "....Extracting PPG from ${exp_dir}/editing/wav_${test_dir}_$vocoder";
    
    ./ppg_tts/evaluation/evaluate_ppg/extract_kaldi_ppg.sh \
        --wav_dir ${exp_dir}/editing/wav_${test_dir}_$vocoder \
        --text_file ${exp_dir}/editing/text
fi

if [ $start -le 4 ] && [ $end -ge 4 ]; then
    echo "Evaluating PPG between our model and baseline TTS"
    python -m ppg_tts.evaluation.evaluate_ppg.evaluate \
        --tts_baseline_ppg ${exp_dir}/editing/wav_baseline_${test_dir}_$vocoder/kaldi_dataset/ppg.scp \
        --ppg ${exp_dir}/editing/wav_${test_dir}_$vocoder/kaldi_dataset/ppg.scp
fi

echo "Step $start to step $end is done";
