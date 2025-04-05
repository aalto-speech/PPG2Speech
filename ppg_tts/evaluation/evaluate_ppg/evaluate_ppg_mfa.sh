#!/usr/bin/bash

testset=$1
model_class=$2
ckpt=$3
device=$4
vocoder=$5
baseline_tts_proj=$6
rule_based=$7

if [ $# -lt 7 ]; then
    echo "Usage: $0 <testset> <model_class> <ckpt> <device> <vocoder> <baseline_tts_proj> <rule_based> [<guidance> <sway> <start> <end>]"
    exit 1
fi

guidance="${8:-1.0}"
sway="${9:--1.0}"
start="${10:-0}"
end="${11:-5}"

exp_dir=$(realpath $(dirname "$(dirname "$ckpt")"))
test_dir=$(basename ${testset})

if [ "$rule_based" = "--rule_based_edit" ]; then
    flag="_rule_based"
else
    flag=""
fi

set -e
set -o pipefail

if [ $start -le 0 ] && [ $end -ge 0 ]; then
    echo "Step 0: Generating mels with edited PPGs"

    python -m ppg_tts.evaluation.synthesis --model_class ${model_class} \
        --ckpt ${ckpt} --device ${device} --data_dir ${testset} --edit_ppg \
        --guidance ${guidance} --sway_coeff ${sway}
fi

if [ $start -le 1 ] && [ $end -ge 1 ]; then
    echo "Step 1: Generating wavs for edited mels"

    if [[ $vocoder == "bigvgan" ]]; then

        curr_dir=$(pwd)

        cd vocoder/bigvgan
        python inference_e2e.py --checkpoint_file bigvgan_generator.pt \
            --input_mels_dir ${exp_dir}/editing_${test_dir}${flag}/mel_gd${guidance}_sw${sway} \
            --output_dir ${exp_dir[$SLURM_ARRAY_TASK_ID]}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}

        cd $curr_dir
    else
        python -m vocoder.hifigan.inference_e2e \
            --checkpoint_file vocoder/hifigan/ckpt/g_02500000 \
            --input_mels_dir ${exp_dir}/editing_${test_dir}${flag}/mel_gd${guidance}_sw${sway} \
            --output_dir ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}
    fi
    cp ${exp_dir}/editing_${test_dir}${flag}/mel_gd${guidance}_sw${sway}/speaker_mapping \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/speaker_mapping
fi


if [ $start -le 2 ] && [ $end -ge 2 ]; then
    echo "Step 2: Inference TTS-baseline for edited text";

    curr_dir=$(pwd)
    
    cd ${baseline_tts_proj}

    echo "Start Inference for the model on ${testset}"

    sbatch --wait --output=${exp_dir}/editing_${test_dir}${flag}/baseline_tts_infer_%A.out \
        ${baseline_tts_proj}/sbatch_scripts/inference.sh \
        ${exp_dir}/editing_${test_dir}${flag}/text \
        ${testset}/embedding.scp \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway} \
        ${guidance} ${sway}

    cd $curr_dir

    awk '{print $1 " " $1}' ${exp_dir}/editing_${test_dir}${flag}/text \
        > ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/speaker_mapping
fi

if [ $start -le 3 ] && [ $end -ge 3 ]; then
    echo "Step 3: Extract alignment from baseline TTS using MFA";

    echo "....Make corpus and dictionary for MFA";
    python ppg_tts/evaluation/evaluate_ppg/build_mfa_corpus.py \
        --text_file ${exp_dir}/editing_${test_dir}${flag}/text \
        --audio_dir ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway} \
        --output_dir ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus \

    python ppg_tts/evaluation/evaluate_ppg/build_mfa_dictionary.py \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/dictionary;

    python ppg_tts/evaluation/evaluate_ppg/build_mfa_corpus.py \
        --text_file ${exp_dir}/editing_${test_dir}${flag}/text \
        --audio_dir ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway} \
        --output_dir ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus \

    python ppg_tts/evaluation/evaluate_ppg/build_mfa_dictionary.py \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/dictionary;

    mv ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/03m/03m_test_0001.lab \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/03m/03m_test_0001.lab.tmp;
    mv ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/03m/03m_test_0002.lab \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/03m/03m_test_0001.lab;
    mv ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/03m/03m_test_0001.lab.tmp \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/03m/03m_test_0002.lab;
    
    echo "Extract alignment using MFA";
    sbatch --wait --output=${exp_dir}/editing_${test_dir}${flag}/mfa_basetts_%A.out \
        ppg_tts/evaluation/evaluate_ppg/mfa_align.sh \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/dictionary \
        /scratch/elec/t412-speechsynth/DATA/fin-mix/mfa_model/perso.zip \
        ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_align;

    sbatch --wait --output=${exp_dir}/editing_${test_dir}${flag}/mfa_ppg2speech_%A.out \
        ppg_tts/evaluation/evaluate_ppg/mfa_align.sh \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_corpus/dictionary \
        /scratch/elec/t412-speechsynth/DATA/fin-mix/mfa_model/perso.zip \
        ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_align;
fi

if [ $start -le 4 ] && [ $end -ge 4 ]; then
    echo "Step 4: Extract PPG from synthesized speech"

    echo "....Extracting PPG from ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}";

    ./ppg_tts/evaluation/evaluate_ppg/extract_kaldi_ppg.sh \
        --wav_dir ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway} \
        --text_file ${exp_dir}/editing_${test_dir}${flag}/text

    echo "....Extracting PPG from ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}";
    
    ./ppg_tts/evaluation/evaluate_ppg/extract_kaldi_ppg.sh \
        --wav_dir ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway} \
        --text_file ${exp_dir}/editing_${test_dir}${flag}/text
fi

if [ $start -le 5 ] && [ $end -ge 5 ]; then
    echo "Step 5: Evaluating PPG between our model and baseline TTS";

    echo "Evaluating PPG from model synthesized speech";
    python -m ppg_tts.evaluation.evaluate_ppg.evaluate \
        --edited_ppg ${exp_dir}/editing_${test_dir}${flag}/ppg.scp \
        --synthesized_ppg ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/kaldi_dataset/ppg.scp \
        --edit_json ${exp_dir}/editing_${test_dir}${flag}/edits.json \
        --matcha_mfa_align ${exp_dir}/editing_${test_dir}${flag}/wav_${vocoder}_gd${guidance}_sw${sway}/mfa_align

    echo " ";
    echo "Evaluating PPG from TTS-baseline synthesized speech";
    python -m ppg_tts.evaluation.evaluate_ppg.evaluate \
        --edited_ppg ${exp_dir}/editing_${test_dir}${flag}/ppg.scp \
        --synthesized_ppg ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/kaldi_dataset/ppg.scp \
        --edit_json ${exp_dir}/editing_${test_dir}${flag}/edits.json \
        --matcha_mfa_align ${exp_dir}/editing_${test_dir}${flag}/wav_baseline_${vocoder}_gd${guidance}_sw${sway}/mfa_align
fi

echo "Step $start to step $end is done";
