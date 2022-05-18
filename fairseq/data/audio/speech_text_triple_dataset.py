# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)

from fairseq.data.audio.speech_to_text_dataset import (
    get_features_or_waveform,
    _collate_frames,
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator
)

logger = logging.getLogger(__name__)


class SpeechTextTripleDataset(SpeechToTextDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
            self,
            split: str,
            is_train_split: bool,
            data_cfg: S2TDataConfig,
            audio_paths: List[str],
            n_frames: List[int],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            tgt_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
    ):
        super().__init__(split, is_train_split,
                         data_cfg, audio_paths, n_frames,
                         src_texts, tgt_texts, speakers, src_langs, tgt_langs,
                         ids, tgt_dict, pre_tokenizer, bpe_tokenizer)
        self.dataset_type = "st" # default
        if "mt" in split:
            self.dataset_type = "mt"
        self.check_src_lang_tag()

    def check_src_lang_tag(self):
        if self.data_cfg.prepend_src_lang_tag:
            assert self.src_langs is not None and self.tgt_dict is not None
            src_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.src_langs)
            ]
            assert all(t in self.tgt_dict for t in src_lang_tags)

    def __getitem__(
            self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        audio = None
        if self.dataset_type == "st":
            audio = get_features_or_waveform(
                self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
            )
            if self.feature_transforms is not None:
                assert not self.data_cfg.use_audio_input
                audio = self.feature_transforms(audio)
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            if self.data_cfg.use_audio_input:
                audio = audio.squeeze(0)

        src_text = None
        if self.src_texts is not None:
            tokenized = self.tokenize_text(self.src_texts[index])
            src_text = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_src_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                src_text = torch.cat((torch.LongTensor([lang_tag_idx]), src_text), 0)

        tgt_text = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            tgt_text = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                tgt_text = torch.cat((torch.LongTensor([lang_tag_idx]), tgt_text), 0)

        return index, audio, src_text, tgt_text

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _ in samples], dtype=torch.long)
        if self.dataset_type == "st":
            frames = _collate_frames(
                [s for _, s, _, _ in samples], self.data_cfg.use_audio_input
            )
            # sort samples by descending number of frames
            n_frames = torch.tensor([s.size(0) for _, s, _, _ in samples], dtype=torch.long)
            n_frames, order = n_frames.sort(descending=True)
            indices = indices.index_select(0, order)
            frames = frames.index_select(0, order)
        else:
            frames, n_frames = None, None
            order = indices
        # process source text
        source, source_lengths = None, None
        prev_output_source_tokens = None
        src_ntokens = None
        if self.src_texts is not None:
            source = fairseq_data_utils.collate_tokens(
                [s for _, _, s, _ in samples],
                self.tgt_dict.pad(), self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )

            if self.dataset_type == "mt":
                source_lengths = torch.tensor([s.size(0) for _, _, s, _ in samples], dtype=torch.long)
                source_lengths, order = source_lengths.sort(descending=True)

            source = source.index_select(0, order)
            if self.dataset_type == "st":
                source_lengths = torch.tensor(
                    [s.size() for _, _, s, _ in samples], dtype=torch.long
                ).index_select(0, order)
            src_ntokens = sum(s.size(0) for _, _, s, _ in samples)
            prev_output_source_tokens = fairseq_data_utils.collate_tokens(
                [s for _, _, s, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_source_tokens = prev_output_source_tokens.index_select(0, order)
        # process target text
        target, target_lengths = None, None
        prev_output_target_tokens = None
        tgt_ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, _, t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_target_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_target_tokens = prev_output_target_tokens.index_select(0, order)
            tgt_ntokens = sum(t.size(0) for _, _, _, t in samples)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_target_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "target_ntokens": tgt_ntokens,
            "nsentences": len(samples),
            "source": source,
            "source_lengths": source_lengths,
            "source_ntokens": src_ntokens,
            "prev_output_src_tokens": prev_output_source_tokens,
            "dataset_type": self.dataset_type
        }
        return out


class SpeechTextTripleDatasetCreator(SpeechToTextDatasetCreator):

    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[List[Dict]],
            data_cfg: S2TDataConfig,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
    ) -> SpeechTextTripleDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss.get(cls.KEY_ID, None) for ss in s])
            audio_paths.extend(
                [os.path.join(data_cfg.audio_root, ss.get(cls.KEY_AUDIO, "")) for ss in s]
            )
            n_frames.extend([int(ss.get(cls.KEY_N_FRAMES, 0)) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        return SpeechTextTripleDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )

    @classmethod
    def from_tsv(
            cls,
            root: str,
            data_cfg: S2TDataConfig,
            splits: str,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split: bool,
            epoch: int,
            seed: int,
    ) -> SpeechTextTripleDataset:
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = os.path.join(root, f"{split}.tsv")
            if not os.path.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(f, delimiter="\t", quotechar=None,
                                        doublequote=False, lineterminator="\n",
                                        quoting=csv.QUOTE_NONE)
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
