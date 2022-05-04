# Copyright (c) ByteDance, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("label_smoothed_cross_entropy_with_constrastive")
class LabelSmoothedCrossEntropyWithContrastiveCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        contrastive_weight=0.0,
        contrastive_temperature=1.0,
        use_dual_ctr=False,
        ctr_dropout_rate=0.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        self.use_dual_ctr = use_dual_ctr
        self.ctr_dropout_rate = ctr_dropout_rate

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--contrastive-weight', default=0., type=float,
                            help='the weight of contrastive loss')
        parser.add_argument('--contrastive-temperature', default=1.0, type=float,
                            help='the temperature in the contrastive loss')
        parser.add_argument('--contrastive-seqlen-type', default='src_text', type=str,
                            choices=['src_text', 'transcript',
                                     'audio_short', 'none'],
                            help='which type of length to times to the contrastive loss')

        parser.add_argument("--use-dual-ctr", action="store_true",
                            help="if we want to use dual contrastive loss")
        parser.add_argument("--ctr-dropout-rate", default=0., type=float,
                            help='the dropout rate of hidden units')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        _net_output = model(**sample["net_input"]) # (x, extra)
        if model.training:
            net_output, encoder_out = _net_output
            contrastive_loss, short_audio_len = self.compute_contrastive_loss(
                model, sample, encoder_out,
                reduce=reduce, return_short_audio_len=True
            )
        else:
            net_output = _net_output
            contrastive_loss, short_audio_len = torch.tensor(0.0), None
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        if sample["target"] is not None: # ST triple dataset
            label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        )
        source_ntokens = sample["source_ntokens"]
        if label_smoothed_nll_loss is not None:
            loss = label_smoothed_nll_loss + self.contrastive_weight * contrastive_loss
        else:
            loss = contrastive_loss

        logging_output = {
            "loss": loss.data,
            "label_smoothed_nll_loss": label_smoothed_nll_loss.data,
            "nll_loss": nll_loss.data,
            "contrastive_loss": contrastive_loss.data,
            "source_ntokens": source_ntokens,
            "target_ntokens": sample["target_ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if nll_loss != 0:
            logging_output["ntokens"] = sample["target_ntokens"]

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_sequence_hidden(self, model, sample, packed_encoder_out,
                            is_text=False, return_short_audio_len=False):
        short_audio_len = None
        if is_text:
            encoder_out, encoder_padding_mask = model.encoder.embedding_text(
                sample["source"], sample["source_lengths"])
        else:
            encoder_out = packed_encoder_out.encoder_embedding
            encoder_padding_mask = packed_encoder_out.encoder_padding_mask
            short_audio_len = packed_encoder_out.output_encoder_lengths
        encoder_out = encoder_out.transpose(0, 1) # T x B x hid -> B x T x hid
        encoder_padding_mask = (~encoder_padding_mask).float()
        seq_hidden = (encoder_out * encoder_padding_mask.unsqueeze(-1)).sum(dim=1) / encoder_padding_mask.sum(dim=1).unsqueeze(-1)
        return seq_hidden, short_audio_len

    def compute_contrastive_loss(self, model, sample, encoder_out,
                                 reduce=True, return_short_audio_len=True):
        audio_seq_hidden, short_audio_len = self.get_sequence_hidden(model, sample, encoder_out,
                                                                     is_text=False,
                                                                     return_short_audio_len=return_short_audio_len) # B x h

        text_seq_hidden, _ = self.get_sequence_hidden(model, sample, encoder_out, is_text=True) # B x h
        batch_size, hidden_size = audio_seq_hidden.size()
        logits = F.cosine_similarity(audio_seq_hidden.expand((batch_size, batch_size, hidden_size)),
                                     text_seq_hidden.expand((batch_size, batch_size, hidden_size)).transpose(0, 1),
                                     dim=-1)
        logits /= self.contrastive_temperature

        if self.use_dual_ctr:
            loss_audio = -torch.nn.LogSoftmax(0)(logits).diag()
            loss_text = -torch.nn.LogSoftmax(1)(logits).diag()
            loss = loss_audio + loss_text
        else:
            loss = -torch.nn.LogSoftmax(0)(logits).diag()

        if reduce:
            loss = loss.sum()
        return loss, short_audio_len

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        label_smoothed_nll_loss_sum = sum(log.get("label_smoothed_nll_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        target_ntokens = sum(log.get("target_ntokens", 0) for log in logging_outputs)
        source_ntokens = sum(log.get("source_ntokens", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "label_smoothed_nll_loss", label_smoothed_nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / target_ntokens / math.log(2), target_ntokens, round=3
        )
        metrics.log_scalar(
            "contrasitve_loss", contrastive_loss_sum / nsentences / math.log(2), nsentences, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )