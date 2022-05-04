# ConST: Cross-modal Contrastive Learning for Speech Translation 


This is an implementation of NAACL 2022 paper "Cross-modal Contrastive Learning for Speech Translation ". 
The implementation based on [fairseq](https://github.com/pytorch/fairseq) codebase.

**CONTRIBUTION:**
You are also more than welcomed to test our code on your machines, and report feedbacks on results, bugs and performance!

## Introduction
The motivation of the contrastive method is to learn similar representations for semantically similar speech and text.
<div align="center">
  <img src="ConST/resources/motivation_figure.png" width="100%">
</div>

ConST method:
<div align="center">
  <img src="ConST/resources/ConST_figure.png" width="100%">
</div>

### Result on MuST-C En-X dataset
We report **case-sensitive detokenized BLEU** via sacrebleu toolkit.

| Model      | En-De | En-Es | En-Fr | En-It | En-Nl | En-Pt | En-Ro | En-Ru | Avg.  |
| ---------- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|ConST-base  |	25.7 |  30.4 |  36.8 | 26.3  |  30.6 |  32.0 | 24.8  |  17.3 |  28.0 |
|ConST-expand|  28.3 |  32.0 |  38.3 | 27.2  |  31.7 |  33.1 | 25.6  |  18.9 |  29.4 |



## Requirements and Installation
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
```bash
git clone git@github.com:ReneeYe/ConST.git
cd ConST
pip3 install -r requirements.txt
python3 setup.py install
python3 setup.py build_ext --inplace
```

## Pre-processing and Training
The instructions of data pre-processing are [here](ConST/prepare_data/README.md).
To train the model, take En-De as an example, you may run:
```bash
bash ConST/scripts/train_en2x.sh de checkpoint/model_saved.
```

## Citation
```
@InProceedings{ye2022cross,
  author    = {Rong Ye and Mingxuan Wang and Lei Li},
  booktitle = {Proc. of NAACL},
  title     = {Cross-modal Contrastive Learning for Speech Translation },
  year      = {2022}
}
```
