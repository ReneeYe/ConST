#!/usr/bin/env bash

set -e

TEXT_PATH=$1
VOCAB_PREFIX=$2
TGT_LANG=$3
shift 3

python3 setup.py install
python3 setup.py build_ext --inplace

spm_model=${VOCAB_PREFIX}.model
spm_dict=${VOCAB_PREFIX}}.txt
# train data
python3 ConST/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/train.${TGT_LANG} --output-file ${TEXT_PATH}/train.spm.${TGT_LANG} --add_lang_tag ${TGT_LANG}
python3 ConST/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/train.en --output-file ${TEXT_PATH}/train.spm.en --add_lang_tag en
# dev data
python3 ConST/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/dev.${TGT_LANG} --output-file ${TEXT_PATH}/dev.spm.${TGT_LANG} --add_lang_tag ${TGT_LANG}
python3 ConST/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/dev.en --output-file ${TEXT_PATH}/dev.spm.en --add_lang_tag en

fairseq-preprocess \
    --source-lang en --target-lang ${TGT_LANG} \
    --trainpref ${TEXT_PATH}/train.spm --validpref ${TEXT_PATH}/dev.spm \
    --destdir ${TEXT_PATH}/bin --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict ${spm_dict} --tgtdict ${spm_dict} \
    --workers 100