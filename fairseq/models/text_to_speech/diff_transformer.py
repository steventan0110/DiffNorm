# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Callable
import torch
from torch import Tensor
from fairseq import checkpoint_utils, utils
import math
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.speech_to_speech.modules.stacked_embedding import StackedEmbedding
from fairseq.models.nat.nonautoregressive_transformer import _mean_pooling
from fairseq.models.transformer import (
    Linear,
    TransformerDecoder,
    TransformerModelBase,
    Embedding,
)
from fairseq.modules import PositionalEmbedding
from fairseq.models.nat import NATransformerModel, FairseqNATDecoder, ensemble_decoder
import torch.nn as nn
from functools import partial
from itertools import repeat
import collections.abc

logger = logging.getLogger(__name__)



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)



class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # first reshape to keep H unchanged (T_y is consistent)


        exit(0)
        # C=8 (latent channel size), H=Ty/4, W=H/4=192
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h)) # pad to right and bottom
        x = self.proj(x) # B, C, H/4, W/8 as I use patch size of (4,8) for width and height axis
        print(x.shape)
        x = x.transpose(1, 2).reshape(B, x.shape[2], -1) # B, Ty, C
        print(x.shape)
        print(x[0, :, :])
        print(x[1, :, :])
        exit(0)
        if self.flatten:
            # be careful of padding here
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # print(t.shape)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, vae_dim, out_channels):
        super().__init__()
        self.vae_dim = vae_dim
        self.out_channels = out_channels
        self.linear = nn.Linear(hidden_size, vae_dim * out_channels , bias=True)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        # )

    def forward(self, x):
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # x = modulate(self.norm_final(x), shift, scale)
        # x: B, Ty, C=768 => B, Ty, (8 x 192) => B, 8, Ty, 192
        x = self.linear(x)
        B, Ty, _ = x.shape
        x = x.reshape(B, self.vae_dim, Ty, self.out_channels)
        return x

class DiffusionTransformerModel(TransformerDecoder):
    """Based on DiT model, but using regular cross-attention instead of AdaLN-Zero."""
    @staticmethod
    def add_args(parser):
        # input
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
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        # speaker
        parser.add_argument(
            "--speaker-embed-dim",
            type=int,
            metavar="N",
            help="speaker embedding dimension",
        )
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        # create a fake embed tokens to avoid init error
        embed_tokens = StackedEmbedding(
            1, args.encoder_embed_dim, 0
        )
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = 1
        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )
        self.ensemble_models = None
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, args.encoder_embed_dim, None)
        # self.length_scorer = nn.Sequential(
        #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(args.encoder_embed_dim, 1)
        # )
        self.noise_flatten = Mlp(192*8, hidden_features=2 * args.encoder_embed_dim, out_features=args.encoder_embed_dim, drop=0.1)
        self.t_embedder = TimestepEmbedder(args.encoder_embed_dim) # embed time step of current diffusion step
        # self.source_mlp = Mlp(args.encoder_embed_dim, hidden_features= 2 * args.encoder_embed_dim, out_features=args.encoder_embed_dim, drop=0.1)
        self.final_layer = FinalLayer(args.encoder_embed_dim, 8, 192)

        self.embed_tokens = None # remove the embed_tokens, as we don't need it
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                192, # 768 / 4, to be consistent with vae feature's dimension
                self.padding_idx,
                learned=False,
            )
        )

        self.initialize_weights()


    def initialize_weights(self):
        nn.init.normal_(self.embed_length.weight, mean=0, std=self.args.encoder_embed_dim ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # init for embedders
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.noise_flatten.fc1.weight, std=0.02)
        nn.init.normal_(self.noise_flatten.fc2.weight, std=0.02)

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        self.output_projection = None
        # Mlp(self.output_embed_dim, hidden_features= 2 * self.output_embed_dim, out_features=self.output_embed_dim, drop=0.1, norm_layer=nn.LayerNorm))


    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tensor] = None,
        src_lengths: Optional[Tensor] = None,
        tgt_lengths: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        normalize = False,
        return_all_hiddens: bool = False,
        diffusion_steps: Optional[int] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # x: B, tgt_len ,256
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            incremental_state=incremental_state,
            full_context_alignment=True, # for nar, decoder should always see all the tokens
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            diffusion_steps=diffusion_steps,
        )
        latent = self.final_layer(x)
        # if not features_only:
        #     x = self.output_projection(x)
        return latent, extra

    def extract_features(
            self,
            prev_output_tokens,
            src_lengths: Optional[Tensor],
            tgt_lengths: Optional[Tensor],
            encoder_out: Optional[Tensor],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            diffusion_steps: Optional[int] = None,
    ):
        """
        prev_output_tokens: B, L, C=768 (as mhubert feature's dim is 768), this is the noised version of target feature
        encoder_out[0]: L x B x C (note that L, B is not the same order as prev_output_tokens)
        diffusion_steps: (B,), diffusion timestep for each instance in the batch
        """
        # B, Tx and B, Ty
        src_padding_mask, tgt_padding_mask = lengths_to_padding_mask(src_lengths), lengths_to_padding_mask(tgt_lengths)

        # src_feature = self.source_mlp(encoder_out) # B, L, C
        # TODO: maybe try DIT's MLP and scale&shift approach
        src_feature = encoder_out # we directly use the source mhubert feature as the conditioning feature
        time_embed =self.t_embedder(diffusion_steps) # B, C
        condition_feature = src_feature + time_embed.unsqueeze(1) # B, L, C; time embed is broadcasted
        # prepare target patches; first inject sinusoidal positional embedding
        pos_emb = self.embed_positions(tgt_padding_mask).unsqueeze(1) # B, 1, Ty, C
        noise_feature = prev_output_tokens + pos_emb # B, z, Ty, H/4
        B, Z, Ty, H = noise_feature.shape
        noise_feature = noise_feature.transpose(1, 2).reshape(B, Ty, -1) # B, Ty, z, H/4
        tgt_seq = self.noise_flatten(noise_feature)

        return self.extract_features_scriptable(
            tgt_seq,
            tgt_lengths,
            condition_feature,
            src_padding_mask,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            tgt_lengths: Optional[Tensor],
            encoder_out: Optional[Dict[str, List[Tensor]]],
            encoder_padding_mask: Optional[Tensor],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        prev_output_tokens: B, L, C, noised version of target feature
        encoder_out: B, L, C; transformed encoder output + time embedding, used as the conditioning feature
        encoder_padding_mask: B, L
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1
        assert incremental_state is None, "we generate non-autoregressively and incremental state should never be used!"
        tgt_padding_mask = lengths_to_padding_mask(tgt_lengths) # B, L

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        x = prev_output_tokens

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        self_attn_padding_mask = tgt_padding_mask
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                encoder_out.transpose(0, 1), # need T, B, C format
                encoder_padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            # not used in our case
            x = self.project_out_dim(x) # B, L, C

        return x, {"attn": [attn], "inner_states": inner_states}


    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]


    def forward_length(self, src_feature, mask, target_lengths):
        # # B x T x C -> B x T x 1 -> B x T
        # length_score = self.length_scorer(src_feature).squeeze(2)
        # alphas = torch.sigmoid(length_score)
        # alphas = alphas * mask
        # # print(alphas.sum(dim=1)[:10])
        # # print(target_lengths[:10])
        src_encoding = _mean_pooling(src_feature.transpose(0,1), mask)
        length_out = F.linear(src_encoding, self.embed_length.weight)
        # quantity_loss = (alphas.sum(dim=1) - target_lengths) ** 2
        # quantity_loss = quantity_loss.mean()
        return length_out

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset: # false
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt



def base_multitask_text_transformer_decoder_arch(args):
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_layers = getattr(args, "decoder_layers", 2)

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)

    # decoder layer
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)

    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)


def base_s2st_transformer_encoder_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512) # change from 512 to 768 as default, to work with mhubert feature dimension
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 256)