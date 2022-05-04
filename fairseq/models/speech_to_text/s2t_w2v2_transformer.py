#!/usr/bin/env python3

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils, tasks
from fairseq.data.data_utils import lengths_to_padding_mask
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
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable, S2TTransformerEncoder
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecCtc


from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("s2t_transformer_w2v2")
class S2TTransformerModelW2V2(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument("--w2v2-model-path", type=str, metavar="N",
                            help="path/to/wav2vec/model, support hdfs")
        parser.add_argument("--freeze-w2v", action="store_true",
                            help="if we want to freeze the w2v features")
        parser.add_argument("--use-asr-finetune-w2v", action="store_true",
                            help="if we want to load wav2vec2.0 asr finetuned data")
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2T_W2V2_TransformerEncoder(args)
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
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
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

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        # print("encoder out is ...",encoder_out)
        # print(encoder_out.encoder_out.size())
        # print(type(encoder_out))
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out._asdict()
        )
        return decoder_out


class S2T_W2V2_TransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input wav2vec2Encoder, subsampler and
    Transformer encoder."""

    def __init__(self, args):
        super().__init__(None)

        assert args.w2v2_model_path is not None
        self.w2v2_model_path = args.w2v2_model_path
        self.use_asr_finetune_w2v = args.use_asr_finetune_w2v

        ckpt = torch.load(self.w2v2_model_path)
        self.w2v_args = ckpt["args"]

        if not self.use_asr_finetune_w2v: # if use ssl-trained only
            self.w2v_args = ckpt["args"]
            self.wav2vec_model = Wav2Vec2Model.build_model(ckpt['args'], task=None)
            self.wav2vec_model.load_state_dict(ckpt['model'])
        else: # wav2vec-ctc model
            ckpt["args"].data = args.data
            if not os.path.exists(os.path.join(ckpt["args"].data, f"dict.{ckpt['args'].labels}.txt")):
                os.system(f"wget -P {ckpt['args'].data} https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt")

            task = tasks.setup_task(ckpt["args"])
            model_finetuned = Wav2VecCtc.build_model(ckpt["args"], task=task)
            model_finetuned.load_state_dict(ckpt['model'])
            self.wav2vec_model = model_finetuned.w2v_encoder.w2v_model
            self.w2v_args = ckpt["args"].w2v_args["model"]

        self.freeze_w2v = args.freeze_w2v
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1
        # w2v_output_dim = 512
        w2v_output_dim = self.w2v_args.encoder_embed_dim
        self.subsample = Conv1dSubsampler(
            w2v_output_dim,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _get_w2v_feature(self, src_tokens, src_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        # print("padding mask:", padding_mask.size())
        # print(padding_mask)
        # w2v_feature = self.wav2vec_model.feature_extractor(src_tokens).transpose(1,2)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(src_tokens, padding_mask)
        # print("after extraction, padding:", padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        # output_length = (torch.ones(padding_mask.size()) - padding_mask.int()).sum(dim=1)

        return w2v_feature, padding_mask, output_length

    def forward(self, src_tokens, src_lengths):
        # 1. wav2vec
        # print(src_tokens.size(), src_lengths.size())
        if self.freeze_w2v:
            with torch.no_grad():
                w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                    src_tokens, src_lengths)
        else:
            w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                src_tokens, src_lengths)

        # 2. conv-layers
        # print("after w2v extract, x:", w2v_feature.size())
        x, input_lengths = self.subsample(w2v_feature, input_lengths)
        # x, input_lengths = self.subsample(src_tokens, src_lengths)
        # print("after convs-2, x", x.size())
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        # x = w2v_feature
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        # print(positions.size())
        x += positions
        x = self.dropout_module(x)

        # 3. Transformer-layers
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """

        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

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

        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@register_model_architecture(model_name="s2t_transformer_w2v2", arch_name="s2t_transformer_w2v2")
def base_architecture(args):
    # Wav2vec2.0 feature-extractor
    args.w2v2_model_path = getattr(args, "w2v2_model_path", "./wav2vec_small_100h.pt")
    args.freeze_w2v = getattr(args, "freeze_w2v", False) # default is false, 'store_true'

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
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
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("s2t_transformer_w2v2", "s2t_transformer_w2v2_s")
def s2t_transformer_w2v2_s(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_transformer_w2v2", "s2t_transformer_w2v2_sp")
def s2t_transformer_w2v2_sp(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_w2v2_s(args)


@register_model_architecture("s2t_transformer_w2v2", "s2t_transformer_w2v2asr_s")
def s2t_transformer_w2v2asr_s(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", True)
    s2t_transformer_w2v2_s(args)
