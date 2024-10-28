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
if [ -e "${out_dir}/text_male" ] || [ -e "${out_dir}/wav_male.scp" ]; then
    echo "Found ${out_dir}/text_male or ${out_dir}/wav_male.scp, removing old data"
    rm -f "${out_dir}/text_male" "${out_dir}/wav_male.scp"
fi

for speaker in ${male}/*m; do
    if [ -d $speaker ]; then
        echo "....Processing $speaker"
        for utt in ${speaker}/prompts/*.utt; do
            key=$(basename $utt .utt);
            awk -v key=${key} '{print key " " $0}' ${utt} >> ${out_dir}/text_male;
            awk -v key=${key} -v audioPath=${male_audio_16k} '{print key " " audioPath "/" key ".wav"}' ${utt} >> ${out_dir}/wav_male.scp;
        done
    fi
done


echo "Processing female dataset in ${female}"
if [ -e "${out_dir}/text_female" ] || [ -e "${out_dir}/wav_female.scp" ]; then
    echo "Found ${out_dir}/text_female or ${out_dir}/wav_female.scp, removing old data"
    rm -f "${out_dir}/text_female" "${out_dir}/wav_female.scp"
fi

for speaker in ${female}/*; do
    if [ -d $speaker ]; then
        echo "....Processing $speaker"
        for utt in ${speaker}/prompts/*.utt; do
            key=$(basename $utt .utt);
            awk -v key=${key} '{print key " " $0}' ${utt} >> "${out_dir}/text_female";
            awk -v key=${key} -v audioPath=${female_audio_16k} '{print key " " audioPath "/" key ".wav"}' ${utt} >> "${out_dir}/wav_female.scp";
        done
    fi
done

echo "Merge male and female data"
if [ -e "${out_dir}/text" ] || [ -e "${out_dir}/wav.scp" ]; then
    echo "Found ${out_dir}/text or ${out_dir}/wav.scp, removing old data"
    rm -f "${out_dir}/text" "${out_dir}/wav.scp"
fi

cat ${out_dir}/text_male ${out_dir}/text_female > ${out_dir}/text
cat ${out_dir}/wav_male.scp ${out_dir}/wav_female.scp > ${out_dir}/wav.scp

echo "Split train-dev-test set [Not implement yet]"