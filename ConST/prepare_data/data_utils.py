#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os.path as op
from functools import reduce
from typing import Any, Dict, List
import pandas as pd
import sentencepiece as sp

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def gen_vocab(input_path: str,
              output_path_prefix: str,
              model_type="bpe",
              vocab_size=1000,
              accept_language=None,
              user_defined_symbols=None
              ):
    # Train SentencePiece Model
    sp.SentencePieceTrainer.train(input=input_path,
                                  model_prefix=output_path_prefix,
                                  model_type=model_type,
                                  vocab_size=vocab_size,
                                  accept_language=accept_language,
                                  user_defined_symbols=user_defined_symbols,
                                  unk_piece=UNK_TOKEN, unk_id=UNK_TOKEN_ID,
                                  bos_piece=BOS_TOKEN, bos_id=BOS_TOKEN_ID,
                                  eos_piece=EOS_TOKEN, eos_id=EOS_TOKEN_ID,
                                  pad_piece=PAD_TOKEN, pad_id=PAD_TOKEN_ID)
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


def gen_config_yaml(
    data_root,
    spm_filename,
    yaml_filename="config.yaml",
    prepend_tgt_lang_tag=True,
    prepend_src_lang_tag=True
):
    data_root = op.abspath(data_root)
    writer = S2TDataConfigWriter(op.join(data_root, yaml_filename))
    writer.set_audio_root(op.abspath(data_root))
    writer.set_vocab_filename(spm_filename.replace(".model", ".txt"))
    writer.set_input_channels(1)
    writer.set_bpe_tokenizer(
        {
            "bpe": "sentencepiece",
            "sentencepiece_model": op.join(data_root, spm_filename),
        }
    )
    writer.set_prepend_tgt_lang_tag(prepend_tgt_lang_tag)
    writer.set_prepend_src_lang_tag(prepend_src_lang_tag)
    writer.set_use_audio_input(True)
    writer.set_shuffle_dataset(True)
    writer.flush()


def load_df_from_tsv(path: str):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def filter_manifest_df(
    df, is_train_split=False, extra_filters=None, min_n_frames=5, max_n_frames=3000
):
    filters = {
        "no speech": df["audio"] == "",
        f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
        "empty sentence": df["tgt_text"] == "",
    }
    if is_train_split:
        filters[f"long speech (>{max_n_frames} frames)"] = df["n_frames"] > max_n_frames
    if extra_filters is not None:
        filters.update(extra_filters)
    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print(
        "| "
        + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
        + f", total {invalid.sum()} filtered, {valid.sum()} remained."
    )
    return df[valid]


class S2TDataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"
    DEFAULT_INPUT_FEAT_PER_CHANNEL = 80
    DEFAULT_INPUT_CHANNELS = 1

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for S2T data config")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["vocab_filename"] = vocab_filename

    def set_specaugment(
        self,
        time_wrap_w: int,
        freq_mask_n: int,
        freq_mask_f: int,
        time_mask_n: int,
        time_mask_t: int,
        time_mask_p: float,
    ):
        self.config["specaugment"] = {
            "time_wrap_W": time_wrap_w,
            "freq_mask_N": freq_mask_n,
            "freq_mask_F": freq_mask_f,
            "time_mask_N": time_mask_n,
            "time_mask_T": time_mask_t,
            "time_mask_p": time_mask_p,
        }

    def set_specaugment_lb_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=1,
            freq_mask_f=27,
            time_mask_n=1,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_ld_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_sm_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=15,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_specaugment_ss_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_input_channels(self, input_channels=1):
        self.config["input_channels"] = input_channels

    def set_input_feat_per_channel(self, input_feat_per_channel=80):
        self.config["input_feat_per_channel"] = input_feat_per_channel

    def set_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["bpe_tokenizer"] = bpe_tokenizer

    def set_feature_transforms(self, split, transforms: List[str]):
        if "transforms" not in self.config:
            self.config["transforms"] = {}
        self.config["transforms"][split] = transforms

    def set_prepend_tgt_lang_tag(self, flag=True):
        self.config["prepend_tgt_lang_tag"] = flag

    def set_prepend_src_lang_tag(self, flag=True):
        self.config["prepend_src_lang_tag"] = flag

    def set_sampling_alpha(self, sampling_alpha=1.0):
        self.config["sampling_alpha"] = sampling_alpha

    def set_use_audio_input(self, flag=True):
        self.config["use_audio_input"] = flag

    def set_shuffle_dataset(self, flag=True):
        self.config["shuffle"] = flag