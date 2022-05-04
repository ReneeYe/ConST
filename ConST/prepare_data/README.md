# Speech-to-text data preparation

## MuST-C Data Preparation

First, download the raw data from https://ict.fbk.eu/must-c/, unzip the file and save files to path ```${DATA_PATH}```. 
Let's take the En-De translation as an example:
```bash
export DATA_PATH="data"
tar -zxf ${DATA_PATH}/MUSTC_v1.0_en-de.tar.gz -C ${DATA_PATH}
```
Then run the following script to generate the yaml configuration file, tsv file, sub-word model and dictionary. 
In this work, We jointly tokenize the bilingual text (En & X) using [SentencePiece](https://github.com/google/sentencepiece), with a
vocabulary size of 10k. For example,
```bash
python3 ConST/prepare_data/prep_mustc_data.py --data-root ${DATA_PATH} --lang de --vocab-type unigram --vocab-size 10000
```

You can get the directory like this,
```
data
├── en-de
│   └── *.wav of train/dev/tst-COMMON...
├── config_st.yaml
├── dev_st.tsv
├── spm_unigram10000_st.model
├── spm_unigram10000_st.txt
├── spm_unigram10000_st.vocab
├── train_st.tsv
├── tst-COMMON_st.tsv
├── tst-HE_st.tsv
└── MUSTC_v1.0_en-de.tar.gz
```

## [optional] When using extra MT data
We also provide script to pre-process the extra MT data, 
and save the parallel data under ```${EXT_MT_DATA}``` (e.g.```./data/extra_mt```). 
```
data
├── en-de
│   └── *.wav of train/dev/tst-COMMON...
├── extra_mt
│   ├── dev.en
│   ├── dev.de
│   ├── train.en
│   └── train.de
├── spm_unigram10000_st.model
├── spm_unigram10000_st.txt
└── ...
```
Apply the SentencePiece model, which is learned from MuST-C data, to the text files, and make binary data.
Take En-De as an example,
```bash
bash ConST/scripts/prepare_extra_mt.sh data/extra_mt data/spm_unigram10000_st de
```
You can find all the binarized MT data under ```data/extra_mt/bin```.

## Vocabualry for download

TODO