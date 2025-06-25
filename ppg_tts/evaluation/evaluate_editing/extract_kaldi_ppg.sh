#!/usr/bin/bash

wav_dir=
text_file=

stage=1
stop_stage=100
graph_dir=
model_dir=$(dirname $graph_dir)
transition_id_cnt=
nj=4

# Get the directory where the script was called from
current_dir=$(pwd)

# Set Kaldi root
export KALDI_ROOT="/Specify/Your/Kaldi/Root/Here"
KALDI_S5=$KALDI_ROOT/s5

# Parse command-line options
. $KALDI_S5/utils/parse_options.sh

# Convert wav_dir and text_file to absolute paths before changing directory
wav_dir=$(realpath "$wav_dir")
text_file=$(realpath "$text_file")

# Change to the Kaldi s5 directory
cd $KALDI_S5 || { echo "Error: Failed to change to Kaldi s5 directory"; exit 1; }

# Source Kaldi environment scripts
[ -f path.sh ] && . path.sh
[ -f cmd.sh ] && . cmd.sh

set -eu  # Exit immediately if a command fails

if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
    echo "Build dataset and compute features for ${wav_dir}"
    
    ${current_dir}/ppg_tts/evaluation/evaluate_ppg/make_kaldi_dataset.sh ${wav_dir} ${text_file}
    
    utils/utt2spk_to_spk2utt.pl ${wav_dir}/kaldi_dataset/utt2spk > ${wav_dir}/kaldi_dataset/spk2utt
    
    utils/fix_data_dir.sh ${wav_dir}/kaldi_dataset
    
    steps/make_mfcc.sh --mfcc_config conf/mfcc_hires.conf \
        --cmd "$basic_cmd" --nj ${nj} ${wav_dir}/kaldi_dataset
    
    steps/compute_cmvn_stats.sh ${wav_dir}/kaldi_dataset
fi

if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
    echo "Extract frame-level phone prob"
    
    if [[ ! -f $model_dir/final.xent_output.mdl ]]; then
        echo "Generate HMM-DNN model with xent-output"
        nnet3-copy --edits='remove-output-nodes name=output; rename-node old-name=output-xent new-name=output' \
        $model_dir/final.mdl $model_dir/final.xent_output.mdl
    fi

    $nn_decode_cmd ${wav_dir}/kaldi_dataset/log/feats2post.log \
        nnet3-compute --use-gpu=no $model_dir/final.xent_output.mdl \
            scp:${wav_dir}/kaldi_dataset/feats.scp ark:- \| \
        logprob-to-post ark:- ark:- \| \
        post-to-phone-post --transition-id-counts=$transition_id_cnt \
            $model_dir/final.mdl ark:- ark:- \| \
        post-to-feats --post-dim=$(wc -l < ${graph_dir}/phones.txt) \
            ark:- ark:${wav_dir}/kaldi_dataset/ppg_135.ark

    module load mamba
    source activate tts_env
    echo "PPG extraction done. Start post-processing"
    python ${current_dir}/ppg_tts/evaluation/evaluate_ppg/ppg_post_process.py \
        ${graph_dir}/phones.txt ${wav_dir}/kaldi_dataset/ppg_135.ark
fi

# Return to the original directory
cd "$current_dir"
