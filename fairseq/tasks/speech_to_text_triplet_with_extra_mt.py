# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json
import os.path as op
from argparse import Namespace

import torch
import numpy as np

from fairseq import utils, metrics
from fairseq.data import Dictionary, encoders, iterators
from fairseq.data.audio.speech_text_triple_dataset import (
    SpeechTextTripleDataset,
    SpeechTextTripleDatasetCreator
)
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    get_features_or_waveform,
)
from fairseq.data.multi_modality_dataset import ModalityDatasetItem, MultiModalityDataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation import load_langpair_dataset


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4


@register_task("speech_to_text_triplet_with_extra_mt")
class SpeechToTextTripletWithExtraMTTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument("--config-yaml", type=str, default="config.yaml",
                            help="Configuration YAML filename (absolute path)")
        parser.add_argument("--max-audio-tokens", default=1000000, type=int, metavar="N",
                            help="max batch of tokens in audio sequences"),
        parser.add_argument("--max-text-tokens", default=4000, type=int, metavar="N",
                            help="max batch of tokens in text sequences"),
        parser.add_argument("--max-audio-positions", default=6000, type=int, metavar="N",
                            help="max number of tokens in the source audio sequence")
        parser.add_argument("--max-source-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the source text sequence")
        parser.add_argument("--max-target-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the target sequence")
        parser.add_argument("--langpairs", default=None, metavar="S",
                            help='language pairs for text training, eg: `en-de`')
        parser.add_argument('--lang-prefix-tok', default=None, type=str,
                            help="starting token in decoder, eg: `<lang:de>`")
        parser.add_argument("--external-parallel-mt-data", default=None, type=str,
                            help="path to the external parallel mt data, tsv file")
        parser.add_argument("--text-data-sample-ratio", default=1.0, type=float,
                            help="define MT data sample ratio in one batch")

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--eval-bleu-bpe', type=str, metavar='BPE',
                            default=None,
                            help='args for building the bpe, if needed')
        parser.add_argument('--eval-bleu-bpe-path', type=str, metavar='BPE',
                            help='args for building the bpe, if needed')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}")
        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        if args.external_parallel_mt_data is not None:
            if not op.isabs(args.external_parallel_mt_data):
                args.external_parallel_mt_data = op.join(args.data, args.external_parallel_mt_data)
        if args.langpairs is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        if self.data_cfg.prepend_src_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "source language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_langpair_dataset(self):
        split = "train"
        src, tgt = self.args.langpairs.split("-")
        text_dataset = load_langpair_dataset(
            self.args.external_parallel_mt_data,
            split,
            src,
            self.tgt_dict,
            tgt,
            self.tgt_dict,
            combine=False,
            dataset_impl=None,
            upsample_primary=1,
            left_pad_source=False,
            left_pad_target=False,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=False,
            truncate_source=False,
            shuffle=True,
        )
        return text_dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        st_dataset = SpeechTextTripleDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed
        )
        text_dataset = None
        if self.args.external_parallel_mt_data is not None and is_train_split:
            text_dataset = self.load_langpair_dataset()
        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "speech_to_text",
                    st_dataset,
                    [self.args.max_audio_positions, self.args.max_target_positions],
                    self.args.max_audio_tokens,
                    self.args.batch_size
                ),
                ModalityDatasetItem(
                    "text_to_text",
                    text_dataset,
                    [self.args.max_source_positions, self.args.max_target_positions],
                    self.args.max_text_tokens,
                    self.args.batch_size
                )
            ]
            self.datasets[split] = MultiModalityDataset(mdsets)
        else:
            self.datasets[split] = st_dataset

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        if not isinstance(dataset, MultiModalityDataset):
            return super(SpeechToTextTripletWithExtraMTTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache
            )
        assert isinstance(dataset, MultiModalityDataset)
        assert len(dataset.datasets) == 2
        dataset.set_epoch(epoch)
        batch_samplers = dataset.get_batch_samplers([1.0, self.args.text_data_sample_ratio],
                                                    required_batch_size_multiple,
                                                    seed)
        # return a reusable, sharded iterator
        epoch_iter = iterators.GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1,
            # mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            buffer_size=data_buffer_size,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_audio_positions, self.args.max_source_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        model = super(SpeechToTextTripletWithExtraMTTask, self).build_model(args)

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args))
            if args.eval_bleu_bpe is None:
                self.bpe = None
            else:
                self.bpe = self.build_bpe(self.args)

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))

        return model

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechTextTripleDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechTextTripleDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                if key in logging_outputs[0]:
                    return sum(log[key].cpu().numpy() for log in logging_outputs)
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if self.args.lang_prefix_tok is None:
            prefix_tokens = None
        else:
            prefix_tokens = self.tgt_dict.index(self.args.lang_prefix_tok)
            assert prefix_tokens != self.tgt_dict.unk_index
        with torch.no_grad():
            net_input = sample["net_input"]
            if "src_tokens" in net_input:
                src_tokens = net_input["src_tokens"]
            else:
                raise Exception("net_input must have `src_tokens`.")
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
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
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
