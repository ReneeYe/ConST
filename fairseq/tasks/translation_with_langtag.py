# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace

import torch
import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    LanguagePairDataset,
    TransformEosLangPairDataset,
    encoders,
)
from fairseq.data.audio.speech_text_triple_dataset import  SpeechTextTripleDataset
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import load_langpair_dataset, TranslationTask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


@register_task("translation_with_langtag")
class TranslationTaskWithLangtag(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--lang-prefix-tok', default=None, type=str,
                            help="starting token in decoder")

    def build_model(self, args):
        model = super().build_model(args)
        if hasattr(model, "set_mt_only"):
            model.set_mt_only()

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )
            if args.eval_bleu_bpe is None:
                self.bpe = None
            else:
                self.bpe = encoders.build_bpe(
                    Namespace(
                        bpe=args.eval_bleu_bpe,
                        sentencepiece_model=args.eval_bleu_bpe_path
                    )
                )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        text_dataset = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )
        self.datasets[split] = text_dataset

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if hasattr(models[0], "set_mt_only"):
            new_models = []
            for model in models:
                model.set_mt_only()
                new_models.append(model)
            models = new_models

        if self.args.lang_prefix_tok is None:
            prefix_tokens = None
        else:
            prefix_tokens = self.tgt_dict.index(self.args.lang_prefix_tok)
            assert prefix_tokens != self.tgt_dict.unk_index
        with torch.no_grad():
            net_input = sample["net_input"]
            if "src_tokens" in net_input:
                src_tokens = net_input["src_tokens"]
            elif "source" in net_input:
                src_tokens = net_input["source"]
            else:
                raise Exception("net_input must have `src_tokens` or `source`.")
            bsz, _ = src_tokens.size()[:2]

            if prefix_tokens is not None:
                if isinstance(prefix_tokens, int):
                    prefix_tokens = torch.LongTensor([prefix_tokens]).unsqueeze(1)
                    prefix_tokens = prefix_tokens.expand(bsz, -1).to(src_tokens.device)

            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints
            )

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe is not None:
                s = self.bpe.decode(s)
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                    utils.strip_pad(sample["target"][i][1:], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            if self.args.lang_prefix_tok is not None:
                hyp = hyp.replace(self.args.lang_prefix_tok, "")
                ref = ref.replace(self.args.lang_prefix_tok, "")
            hyps.append(hyp)
            refs.append(ref)

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def build_bpe(self, args):
        # ignore args, no one is really using it
        logger.info(f"tokenizer: {self.bpe}")
        return self.bpe