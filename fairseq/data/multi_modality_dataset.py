# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, NamedTuple

import numpy as np
import math
from fairseq.data import ConcatDataset, Dictionary, data_utils
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.data.audio.speech_text_triple_dataset import SpeechTextTripleDatasetCreator

logger = logging.getLogger(__name__)


class ModalityDatasetItem(NamedTuple):
    datasetname: str
    dataset: any
    max_positions: List[int]
    max_tokens: Optional[int] = None
    max_sentences: Optional[int] = None


# MultiModalityDataset: it concate multiple datasets with different modalities.
# Compared with ConcatDataset it can
#   1) sample data given the ratios for different datasets
#   2) it adds mode to indicate what type of the data samples come from.
# It will be used with GroupedEpochBatchIterator together to generate mini-batch with samples
# from the same type of dataset
# If only one dataset is used, it will perform like the original dataset with mode added

class MultiModalityDataset(ConcatDataset):
    def __init__(self, datasets: List[ModalityDatasetItem]):
        id_to_mode = []
        dsets = []
        max_tokens = []
        max_sentences = []
        max_positions = []
        for dset in datasets:
            id_to_mode.append(dset.datasetname)
            dsets.append(dset.dataset)
            max_tokens.append(dset.max_tokens)
            max_positions.append(dset.max_positions)
            max_sentences.append(dset.max_sentences)
        weights = [1.0 for s in dsets]
        super().__init__(dsets, weights)
        self.max_tokens = max_tokens
        self.max_positions = max_positions
        self.max_sentences = max_sentences
        self.id_to_mode = id_to_mode
        self.raw_sub_batch_samplers = []
        self._cur_epoch = 0

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self._cur_epoch = epoch

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        sample = self.datasets[dataset_idx][sample_idx]
        return (dataset_idx, sample)

    def collater(self, samples, **extra_args):
        if len(samples) == 0:
            return {}
        dataset_idx = samples[0][0]
        # make sure all samples in samples are from same dataset
        assert sum([0 if dataset_idx == s[0] else 1 for s in samples]) == 0
        samples = self.datasets[dataset_idx].collater([x[1] for x in samples])
        # add mode
        samples["net_input"]["mode"] = self.id_to_mode[dataset_idx]

        return samples

    def size(self, index: int):
        if len(self.datasets) == 1:
            return self.datasets[0].size(index)
        return super().size(index)

    @property
    def sizes(self):
        if len(self.datasets) == 1:
            return self.datasets[0].sizes
        super().sizes

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        if len(self.datasets) == 1:
            return self.datasets[0].ordered_indices()
        indices_group = []
        for d_idx, ds in enumerate(self.datasets):
            sample_num = self.cumulative_sizes[d_idx]
            if d_idx > 0:
                sample_num = sample_num - self.cumulative_sizes[d_idx - 1]
            assert sample_num == len(ds)
            indices_group.append(ds.ordered_indices())
        return indices_group

    def get_raw_batch_samplers(self, required_batch_size_multiple, seed):
        if len(self.raw_sub_batch_samplers) > 0:
            logger.info(" raw_sub_batch_samplers exists. No action is taken")
            return
        with data_utils.numpy_seed(seed):
            indices = self.ordered_indices()
        for i, ds in enumerate(self.datasets):
            indices[i] = ds.filter_indices_by_size(
                indices[i],
                self.max_positions[i],
            )[0]
            sub_batch_sampler = ds.batch_by_size(
                indices[i],
                max_tokens=self.max_tokens[i],
                max_sentences=self.max_sentences[i],
                required_batch_size_multiple=required_batch_size_multiple,
            )
            self.raw_sub_batch_samplers.append(sub_batch_sampler)

    def get_batch_samplers(self, mult_ratios, required_batch_size_multiple, seed):
        self.get_raw_batch_samplers(required_batch_size_multiple, seed)
        batch_samplers = []
        for i, _ in enumerate(self.datasets):
            if i > 0:
                sub_batch_sampler = [
                    [y + self.cumulative_sizes[i - 1] for y in x]
                    for x in self.raw_sub_batch_samplers[i]
                ]
            else:
                sub_batch_sampler = list(self.raw_sub_batch_samplers[i])
            smp_r = mult_ratios[i]
            if smp_r != 1:
                is_increase = "increased" if smp_r > 1 else "decreased"
                logger.info(
                    "number of batch for the dataset {} is {} from {} to {}".format(
                        self.id_to_mode[i],
                        is_increase,
                        len(sub_batch_sampler),
                        int(len(sub_batch_sampler) * smp_r),
                    )
                )
                mul_samplers = []
                for _ in range(math.floor(smp_r)):
                    mul_samplers = mul_samplers + sub_batch_sampler
                if math.floor(smp_r) != smp_r:
                    with data_utils.numpy_seed(seed + self._cur_epoch):
                        np.random.shuffle(sub_batch_sampler)
                        smp_num = int(
                            (smp_r - math.floor(smp_r)) * len(sub_batch_sampler)
                        )
                    mul_samplers = mul_samplers + sub_batch_sampler[:smp_num]
                sub_batch_sampler = mul_samplers
            else:
                logger.info(
                    "dataset {} batch number is {} ".format(
                        self.id_to_mode[i], len(sub_batch_sampler)
                    )
                )
            batch_samplers.append(sub_batch_sampler)

        return batch_samplers


if __name__ == "__main__":
    data_cfg = S2TDataConfig("data/config_asr-st_with_tag.yaml")
    tgt_dict = Dictionary.load("data/spm_unigram10000_asr-st_with_tag.txt")
    print(tgt_dict)
    print(data_cfg.config)

    from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
    from collections import namedtuple
    from fairseq.tasks.translation import load_langpair_dataset

    text_dataset = load_langpair_dataset(
        "./data/external_wmt/bin_with_tag/",
        "valid",
        "en",
        tgt_dict,
        "de",
        tgt_dict,
        combine=True,
        dataset_impl=None,
        upsample_primary=1,
        left_pad_source=False,
        left_pad_target=False,
        max_source_positions=1024,
        max_target_positions=10424,
        load_alignments=False,
        truncate_source=False,
    )

    bpe_dict = data_cfg.bpe_tokenizer
    BPEInfo = namedtuple('BPEInfo', ['bpe', 'sentencepiece_model'])
    bpe_tokenizer = SentencepieceBPE(BPEInfo(**data_cfg.bpe_tokenizer))

    st_dataset = SpeechTextTripleDatasetCreator.from_tsv(
        # "./data_wmt",
        "./data",
        data_cfg,
        "dev_asr-st_with_tag",
        # "dev_wmt_mt",
        tgt_dict,
        None,
        bpe_tokenizer,
        is_train_split=False,
        epoch=1,
        seed=1)

    mdsets = [
        ModalityDatasetItem(
            "speech_to_text",
            st_dataset,
            [480240, 1024],
            1000000,
            None
        ),
        ModalityDatasetItem(
            "text_to_text",
            text_dataset,
            [1024, 1024],
            4000,
            None
        )
    ]

    dataset = MultiModalityDataset(mdsets)

    batch_sampler = dataset.get_batch_samplers([1, 1], False, 1)

