#!/usr/bin/bash

testset=$1
model_class=$2
ckpt=$3
device=$4
vocoder=$5
baseline_tts_proj=$6

if [ $# -lt 5 ]; then
    echo "Usage: $0 <testset> <model_class> <ckpt> <device> <vocoder> <baseline_tts_proj> [<start> <end>]"
    exit 1
fi

start="${7:-0}"
end="${8:-5}"

exp_dir=$(realpath $(dirname "$(dirname "$ckpt")"))
test_dir=$(basename ${testset})


if [ $start -le 0 ] && [ $end -ge 0 ]; then
    echo "Step 0: Generating mels with flipped speaker identity"

    python -m ppg_tts.evaluation.synthesis --model_class ${model_class} \
        --ckpt ${ckpt} --device ${device} --data_dir ${testset} --edit_ppg
fi

if [ $start -le 1 ] && [ $end -ge 1 ]; then
    echo "Step 1: Generating wavs for edited mels"

    if [[ $vocoder == "bigvgan" ]]; then

        curr_dir=$(pwd)

        cd vocoder/bigvgan
        python inference_e2e.py --checkpoint_file bigvgan_generator.pt \
            --input_mels_dir ${exp_dir}/editing_${test_dir}/mel \
            --output_dir ${exp_dir[$SLURM_ARRAY_TASK_ID]}/editing_${test_dir}/wav_$vocoder

        cd $curr_dir
    else
        python -m vocoder.hifigan.inference_e2e \
            --checkpoint_file vocoder/hifigan/ckpt/g_02500000 \
            --input_mels_dir ${exp_dir}/editing_${test_dir}/mel \
            --output_dir ${exp_dir}/editing_${test_dir}/wav_$vocoder
    fi
    cp ${exp_dir}/editing_${test_dir}/mel/speaker_mapping \
        ${exp_dir}/editing_${test_dir}/wav_$vocoder/speaker_mapping
fi


# if [ $start -le 2 ] && [ $end -ge 2 ]; then
#     echo "Step 2: Inference TTS-baseline for edited text";

#     srun python ppg_tts/evaluation/evaluate_ppg/tts_baseline/inference_matcha.py \
#         ppg_tts/evaluation/evaluate_ppg/tts_baseline/matcha_hifigan.onnx \
#         --file ${exp_dir}/editing/text \
#         --spk ${testset}/embedding.scp \
#         --output-dir ${exp_dir}/editing/wav_baseline_${test_dir}_$vocoder
# fi

if [ $start -le 3 ] && [ $end -ge 3 ]; then
    echo "Step 3: Extract PPG from synthesized speech"

    echo "....Extracting PPG from ${exp_dir}/editing_${test_dir}/wav_baseline_$vocoder";

    ./ppg_tts/evaluation/evaluate_ppg/extract_kaldi_ppg.sh \
        --wav_dir ${exp_dir}/editing_${test_dir}/wav_baseline_$vocoder \
        --text_file ${exp_dir}/editing_${test_dir}/text

    echo "....Extracting PPG from ${exp_dir}/editing_${test_dir}/wav_$vocoder";
    
    ./ppg_tts/evaluation/evaluate_ppg/extract_kaldi_ppg.sh \
        --wav_dir ${exp_dir}/editing_${test_dir}/wav_$vocoder \
        --text_file ${exp_dir}/editing_${test_dir}/text
fi

if [ $start -le 4 ] && [ $end -ge 4 ]; then
    echo "Step 4: Extract duration/alignment from baseline TTS model";

    echo "....Make wavlist for baseline TTS inference";

    python ppg_tts/evaluation/evaluate_ppg/make_audio_filelist.py \
        ${testset}/wav.scp ${exp_dir}/editing_${test_dir}/text ${exp_dir}/editing_${test_dir}/wavlist


    curr_dir=$(pwd)

    echo "....Change directory from ${curr_dir} to ${baseline_tts_proj}";

    cd ${baseline_tts_proj}

    sbatch --wait --output=${exp_dir}/editing_${test_dir}/baseline_tts_align.out \
        ${baseline_tts_proj}/sbatch_scripts/inference.sh \
        ${exp_dir}/editing_${test_dir}/wav_baseline_$vocoder/matcha_duration \
        ${exp_dir}/editing_${test_dir}/wavlist

    echo "....Duration/Alignment extraction done, change back to ${curr_dir}"

    cd ${curr_dir}

    echo "....Transforming baseline TTS duration/alignment to edit_json format"

    python ppg_tts/evaluation/evaluate_ppg/transform_matcha_alignment.py \
        --edit_json ${exp_dir}/editing_${test_dir}/edits.json \
        --matcha_alignment_folder ${exp_dir}/editing_${test_dir}/wav_baseline_$vocoder/matcha_duration \
        --output_json ${exp_dir}/editing_${test_dir}/matcha_edits.json

fi

if [ $start -le 5 ] && [ $end -ge 5 ]; then
    echo "Step 5: Evaluating PPG between our model and baseline TTS";

    echo "Evaluating PPG from model synthesized speech";
    python -m ppg_tts.evaluation.evaluate_ppg.evaluate \
        --edited_ppg ${exp_dir}/editing_${test_dir}/ppg.scp \
        --synthesized_ppg ${exp_dir}/editing_${test_dir}/wav_$vocoder/kaldi_dataset/ppg.scp \
        --edit_json ${exp_dir}/editing_${test_dir}/edits.json

    
    echo "\nEvaluating PPG from TTS-baseline synthesized speech";
    python -m ppg_tts.evaluation.evaluate_ppg.evaluate \
        --edited_ppg ${exp_dir}/editing_${test_dir}/ppg.scp \
        --synthesized_ppg ${exp_dir}/editing_${test_dir}/wav_baseline_$vocoder/kaldi_dataset/ppg.scp \
        --edit_json ${exp_dir}/editing_${test_dir}/edits.json \
        --matcha_aligned_edits ${exp_dir}/editing_${test_dir}/matcha_edits.json
fi

echo "Step $start to step $end is done";
