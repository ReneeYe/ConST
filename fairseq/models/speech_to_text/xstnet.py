#!/usr/bin/env python3

from argparse import Namespace
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils, tasks
from fairseq.data.data_utils import lengths_to_padding_mask, compute_mask_indices
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecCtc
from fairseq.models.speech_to_text.s2t_w2v2_transformer import S2TTransformerModelW2V2

from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("xstnet")
class XSTNet(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.is_text_input = False # default

    @staticmethod
    def add_args(parser):
        S2TTransformerModelW2V2.add_args(parser)
        parser.add_argument("--textual-encoder-embed-dim", type=int, metavar="N",
                            help="encoder embded dim for text input")

    @classmethod
    def build_encoder(cls, args, dict, embed_tokens):
        encoder = XSTNetEncoder(args, dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, dict, embed_tokens):
        return TransformerDecoderScriptable(args, dict, embed_tokens)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        decoder_embed_tokens = cls.build_embedding(args, task.target_dictionary, args.decoder_embed_dim)
        encoder = cls.build_encoder(args, task.target_dictionary, decoder_embed_tokens)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def set_mt_only(self):
        self.is_text_input = True
        self.encoder.is_text_input = True

    def forward(self, src_tokens, src_lengths, prev_output_tokens,
                is_text_input=False, **kwargs):
        if self.is_text_input:
            is_text_input = True
        encoder_out = self.encoder(src_tokens, src_lengths, is_text_input=is_text_input)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens,
                                   encoder_out=encoder_out)
        if self.training:
            return decoder_out, encoder_out
        return decoder_out


class XSTNetEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.args = args
        self.embed_tokens = embed_tokens
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.padding_idx = embed_tokens.padding_idx
        self.textual_encoder_embed_dim = embed_tokens.embedding_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.textual_encoder_embed_dim)

        self._build_acoustic_encoder(args)
        self._build_textual_encoder(args)
        self.is_text_input = False

        # CTC module
        self.use_ctc = (("ctc" in getattr(args, "ablation_type", "")) and (getattr(args, "ablation_weight", 0.0) > 0)) \
            or (("ctc" in getattr(args, "criterion", "")) and (getattr(args, "ctc_weight", 0.0) > 0))
        if self.use_ctc:
            if (getattr(args, "ablation_type", False) == "ctc_cnn") or \
                    (getattr(args, "ctc_type", False) == "ctc_cnn"):
                self.ctc_type = "ctc_cnn"
                self.ctc_projection = nn.Linear(
                    embed_tokens.embedding_dim,
                    embed_tokens.weight.shape[0],
                    bias=False,
                )
                self.ctc_projection.weight = embed_tokens.weight
            elif (getattr(args, "ablation_type", False) == "ctc_w2v") or \
                    (getattr(args, "ctc_type", False) == "ctc_w2v"):
                self.ctc_type = "ctc_w2v"
                self.ctc_projection = nn.Linear(
                    self.w2v_args.encoder_embed_dim,
                    embed_tokens.weight.shape[0],
                )
            self.ctc_softmax = nn.Softmax(dim=-1)

    def _build_acoustic_encoder(self, args):
        assert args.w2v2_model_path is not None
        self.w2v2_model_path = args.w2v2_model_path
        self.use_asr_finetune_w2v = args.use_asr_finetune_w2v
        try:
            ckpt = torch.load(self.w2v2_model_path)
        except FileNotFoundError:
            if not os.path.exists("wav2vec_small.pt"):
                os.system(f"wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt")
            ckpt = torch.load("wav2vec_small.pt")
        self.w2v_args = ckpt["args"]
        if not self.use_asr_finetune_w2v:  # if use ssl-trained only
            self.w2v_args = ckpt["args"]
            self.wav2vec_model = Wav2Vec2Model.build_model(ckpt['args'], task=None)
            self.wav2vec_model.load_state_dict(ckpt['model'])
        else:  # wav2vec-ctc model
            ckpt["args"].data = args.data
            if not os.path.exists(os.path.join(ckpt["args"].data, f"dict.{ckpt['args'].labels}.txt")):
                os.system(f"wget -P {ckpt['args'].data} https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt")
            task = tasks.setup_task(ckpt["args"])
            model_finetuned = Wav2VecCtc.build_model(ckpt["args"], task=task)
            model_finetuned.load_state_dict(ckpt['model'])
            self.wav2vec_model = model_finetuned.w2v_encoder.w2v_model
            self.w2v_args = ckpt["args"].w2v_args["model"]
        self.freeze_w2v = args.freeze_w2v

        w2v_output_dim = self.w2v_args.encoder_embed_dim
        self.subsample_audio = Conv1dSubsampler(
            w2v_output_dim,
            args.conv_channels,
            self.textual_encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

    def _build_textual_encoder(self, args):
        self.max_source_positions = args.max_source_positions
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                self.textual_encoder_embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.textual_encoder_embed_dim)
        else:
            self.layernorm_embedding = None
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.textual_encoder_embed_dim)
        else:
            self.layer_norm = None

    def _get_w2v_feature(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(src_tokens, padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return w2v_feature, padding_mask, output_length

    def embedding_mask_audio_seq(self, src_tokens, src_lengths, mask_configs=None,
                                 return_short_audio_len=False):
        # src_tokens: b x frame, original audio_src_tokens
        padding_mask = lengths_to_padding_mask(src_lengths)
        masked_src_tokens = src_tokens.clone()
        B, T = masked_src_tokens.size()
        if (mask_configs is not None) and (mask_configs.get("mask_seq_prob", 0.0) > 0):
            mask_seq_prob = mask_configs.get("mask_seq_prob", 0.0)
            mask_length = mask_configs.get("mask_length", 3600)
            mask_selection = mask_configs.get("mask_type", "static")
            no_mask_overlap = mask_configs.get("no_mask_overlap", False)
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_seq_prob,
                mask_length,
                mask_selection,
                0.0,
                min_masks=2,
                no_overlap=no_mask_overlap,
                min_space=0
            )
            mask_indices = torch.from_numpy(mask_indices).to(masked_src_tokens.device)
            masked_src_tokens[mask_indices] = 0
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(masked_src_tokens, padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)

        x, output_length = self.subsample_audio(w2v_feature, output_length)
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(output_length)
        if self.embed_positions is not None:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.dropout_module(x)
        if return_short_audio_len:
            return x, encoder_padding_mask, output_length
        return x, encoder_padding_mask, None

    def embedding_audio(self, src_tokens, src_lengths,
                        return_short_audio_len=False):
        if self.freeze_w2v:
            with torch.no_grad():
                w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                    src_tokens, src_lengths)
        else:
            w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                src_tokens, src_lengths)

        x, input_lengths = self.subsample_audio(w2v_feature, input_lengths)
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if self.embed_positions is not None:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.dropout_module(x)
        if return_short_audio_len:
            return x, encoder_padding_mask, input_lengths
        return x, encoder_padding_mask, None

    def embedding_text(self, src_tokens, src_lengths):
        token_embedding = self.embed_tokens(src_tokens)
        x = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        x = x.transpose(0, 1) # B x T x C -> T x B x C
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return x, encoder_padding_mask

    def forward(self, src_tokens, src_lengths, is_text_input=False, **kwargs):
        """
        src_tokens: b x seq, float tensor if it is audio input, LongTensor if it is text input
        src_lengths: b-dim LongTensor
        """
        short_audio_len = None
        if self.is_text_input:
            is_text_input = True
        if is_text_input:
            x, encoder_padding_mask = self.embedding_text(src_tokens, src_lengths)
        else:
            x, encoder_padding_mask, short_audio_len = self.embedding_audio(src_tokens, src_lengths,
                                                                            return_short_audio_len=True)
        encoder_embedding = x
        # 3. Transformer-layers
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=short_audio_len
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """

        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )

        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=None
        )

    def compute_ctc_logit_and_logprob(self, src_tokens, src_lengths):
        assert self.use_ctc, "CTC is not available!"
        w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
            src_tokens, src_lengths
        )
        encoder_state = w2v_feature # b x seq x 768
        if self.ctc_type == "ctc_cnn":
            encoder_state, input_lengths = self.subsample_audio(w2v_feature, input_lengths) # seq x b x 512
            encoder_state = encoder_state.transpose(0, 1) # b x seq x 512
            encoder_state = self.embed_scale * encoder_state
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        else:
            assert self.ctc_type == "ctc_w2v", "ctc type should be ctc_w2v or ctc_cnn"
        encoder_state = self.dropout_module(encoder_state)
        ctc_logit = self.ctc_projection(encoder_state) # b x seq x voc
        logits = ctc_logit.float()
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1) # seq x b x voc
        return ctc_logit, encoder_padding_mask, log_probs


@register_model_architecture(model_name="xstnet", arch_name="xstnet_base")
def base_architecture(args):
    # Wav2vec2.0 feature-extractor
    args.w2v2_model_path = getattr(args, "w2v2_model_path", "./wav2vec_small_100h.pt")
    args.freeze_w2v = getattr(args, "freeze_w2v", False) # default is false, 'store_true'
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_dim = getattr(args, "textual_encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
