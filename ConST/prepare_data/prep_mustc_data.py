#!/usr/bin/env python3
# Copyright (c) ByteDance, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os.path as op
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import torchaudio
from data_utils import filter_manifest_df, gen_config_yaml, gen_vocab, save_df_to_tsv

from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "speaker",
                    "src_text", "tgt_text", "src_lang", "tgt_lang"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = op.join(root, f"en-{lang}", "data", split)
        wav_root, txt_root = op.join(_root, "wav"), op.join(_root, "txt")
        assert op.isdir(_root) and op.isdir(wav_root) and op.isdir(txt_root)
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "the MuST-C dataset")
        with open(op.join(txt_root, f"{split}.yaml")) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(op.join(txt_root, f"{split}.{_lang}")) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = op.join(wav_root, wav_filename)
            relative_wav_path = op.relpath(wav_path, root)
            sample_rate = torchaudio.info(wav_path).sample_rate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{op.splitext(wav_filename)[0]}_{i}"
                self.data.append(
                    (
                        relative_wav_path,
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[str, int, int, int, str, str, str, str]:
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    lang = args.lang
    cur_root = op.join(args.data_root, f"en-{lang}")
    if not op.isdir(cur_root):
        FileExistsError(f"{cur_root} does not exist.")
    train_text = []
    for split in MUSTC.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = MUSTC(args.data_root, lang, split)
        for wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav_path}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(spk_id)
            manifest["src_lang"].append("en")
            manifest["tgt_lang"].append(lang)
            manifest["src_text"].append(src_utt)
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
            train_text.extend(manifest["src_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=1000, max_n_frames=480000)
        save_df_to_tsv(df, op.join(args.data_root, f"{split}_st.tsv"))

    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_st"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            f.name,
            op.join(args.data_root, spm_filename_prefix),
            args.vocab_type,
            args.vocab_size,
            accept_language=["en", f"{lang}"],
            user_defined_symbols=["<lang:en>", f"<lang:{lang}>"],
        )
    # Generate config YAML
    gen_config_yaml(
        args.data_root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_st.yaml",
        prepend_tgt_lang_tag=True,
        prepend_src_lang_tag=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--lang", type=str, default="de",
                        choices=["de", "es", "fr", "it", "nl", "pt", "ro", "ru"])
    parser.add_argument("--vocab-type", default="unigram", required=True, type=str,
                        choices=["bpe", "unigram", "char"])
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
