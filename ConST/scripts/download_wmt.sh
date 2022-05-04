#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

export version="wmt17"
export target=de
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-dir) export DATA_ROOT="$2"; shift ;;
        --external) export external="$2"; shift ;;
        --target) export target="$2"; shift ;;
        --wmt14) export version="wmt14" ;;
        --wmt16) export version="wmt16" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "arguments"
echo "DATA_ROOT: $DATA_ROOT"
echo "version: $version"
echo "target language: $target"
echo

if [[ $version == "wmt16" && $target != "ro" ]] || [[ $version != "wmt16" && $target == "ro" ]]; then
    echo "--wmt16 if and only if target is ro"
    exit
fi

mkdir -p $DATA_ROOT
cd $DATA_ROOT

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [[ "$version" == "wmt14" && $target != "es" ]]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
fi

if [[ "$target" == "ro" ]]; then
    URLS=(
        "http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz"
        "https://opus.nlpl.eu/download.php?f=SETIMES/v2/raw/en.zip"
        "https://opus.nlpl.eu/download.php?f=SETIMES/v2/raw/ro.zip"
        "http://data.statmt.org/wmt16/translation-task/dev.tgz"
        "http://data.statmt.org/wmt16/translation-task/test.tgz"
    )
    FILES=(
        "training-parallel-ep-v8.tgz"
        "en.zip"
        "ro.zip"
        "dev.tgz"
        "test.tgz"
    )
fi

orig=orig
mkdir -p $orig
cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}

    if [[ -f $file ]]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url" -O ${FILES[i]}
        if [[ -f $file ]]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
    fi

    if [[ ${file: -4} == ".tgz"  || ${file: -4} == ".tar" ]]; then
        pigz -dc $file | tar xvf -
    elif [[ ${file: -4} == ".zip" ]]; then
        unzip
    fi
done

if [[ "$target" == "ro" ]]; then
    pair_dir=SETIMES/$src-$tgt
    mkdir -p $pair_dir
    cp SETIMES/raw/$src/setimes.$src-$tgt.xml $pair_dir/setimes.$src-$tgt.$src
    cp SETIMES/raw/$tgt/setimes.$src-$tgt.xml $pair_dir/setimes.$src-$tgt.$tgt
fi