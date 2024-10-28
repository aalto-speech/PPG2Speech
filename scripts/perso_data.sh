#!/usr/bin/bash

data_dir=$1
out_dir=$2

male="${data_dir}/male"
female="${data_dir}/female"

male_audio_16k="${data_dir}/downsampled_16k_male"
female_audio_16k="${data_dir}/downsampled_16k_female"

if [ "$#" -ne 2 ]; then
    echo "Error: Insufficent Arguments. Usage: ./perso_data.sh <data_dir> <out_dir>."
    exit 1
fi

echo "Processing male dataset in ${male}"
for speaker in ${male}/*m; do
    if [ -d $speaker ]; then
        echo "....Processing $speaker"
        for utt in ${speaker}/prompts/*.utt; do
            echo "........Processing ${utt}"
            key=$(basename $utt .utt)
            awk -v key=${key} '{print key " " $0}' # >> ${out_dir}/text_male
            awk -v key=${key} -v audioPath=${male_audio_16k} '{print key " " audioPath "/" key ".wav"}' >> ${out_dir}/wav_male.scp
        done
    fi
done

echo "Processing male dataset in ${female}"
for speaker in ${female}/*; do
    if [ -d $speaker ]; then
        echo "....Processing $speaker"
        for utt in ${speaker}/prompts/*.utt; do
            key=$(basename $utt .utt)
            awk -v key=${key} '{print key " " $0}' # >> ${out_dir}/text_female
            awk -v key=${key} -v audioPath=${female_audio_16k} '{print key " " audioPath "/" key ".wav"}' >> ${out_dir}/wav_female.scp
        done
    fi
done

echo "Merge male and female data"
cat ${out_dir}/text_male ${out_dir}/text_female > ${out_dir}/text
cat ${out_dir}/wav_male.scp ${out_dir}/wav_female.scp > ${out_dir}/wav.scp