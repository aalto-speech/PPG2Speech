#!/usr/bin/bash

testset=$1
ckpt=$2
device=$3
auth_token=$4

if [ $# -lt 4 ]; then
    echo "Usage: $0 <testset> <ckpt> <device> <auth_token>"
    exit 1
fi

exp_dir=$(dirname "$(dirname "$ckpt")")

echo "Generating mels with flipped speaker identity"

python -m ppg_tts.evaluation.switched_spkid_generation --ckpt ${ckpt} --device ${device} --data_dir ${testset}

echo "Generating wavs for flipped generated mels"

source activate hifi-gan

python -m vocoder.hifigan.inference_e2e --checkpoint_file vocoder/hifigan/ckpt/g_02500000 --input_mels_dir ${exp_dir}/flip_generate_mel --output_dir ${exp_dir}/flip_generate_wav

cp ${exp_dir}/flip_generate_mel/speaker_mapping ${exp_dir}/flip_generate_wav/speaker_mapping

echo "Calculate speaker embedding distance between original target speaker wavs and generated target speaker wavs"

source activate tts_env

python -m ppg_tts.evaluation.cal_spk_emb_dist --data_dir ${testset} --flip_wav_dir ${exp_dir}/flip_generate_wav --device ${device}

python -m srun python -m ppg_tts.evaluation.evaluate_wer --data_dir ${testset} --flip_wav_dir ${exp_dir}/flip_generate_wav

python -m ppg_tts.evaluation.dnsmos_eval --flip_wav_dir ${exp_dir}/flip_generate_wav
