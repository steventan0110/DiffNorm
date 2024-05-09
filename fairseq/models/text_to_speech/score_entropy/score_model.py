import math
import copy
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from fairseq.modules import PositionalEmbedding
from fairseq.models.text_to_speech.score_entropy.noise import GeometricNoise, LogLinearNoise, Noise
from fairseq.models.text_to_speech.score_entropy.graph import Uniform, Absorbing
from fairseq.models.text_to_speech.score_entropy.sampling_utils import EulerPredictor, AnalyticPredictor, Denoiser
import numpy as np
from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
)
from fairseq.models.text_to_speech.distributions import DiagonalGaussianDistribution
import sacrebleu
import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from fairseq.models import FairseqEncoder

import torchaudio
import torchaudio.transforms as T

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from typing import Any, List, Tuple, Union, Optional, Dict
# peripheral models

# audio to mel

from collections import namedtuple
from functools import wraps
from packaging import version
from fairseq.utils import new_arange
from torch import nn, einsum


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


mlist = nn.ModuleList


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def divisible_by(num, den):
    return (num % den) == 0


def identity(t, *args, **kwargs):
    return t


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


# tensor helpers

def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t

    if t.shape[-1] > length:
        return t[..., :length]

    return F.pad(t, (0, length - t.shape[-1]))


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def generate_mask_from_repeats(repeats):
    repeats = repeats.int()
    device = repeats.device

    lengths = repeats.sum(dim=-1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim=-1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value=0.)

    seq = torch.arange(max_length, device=device)
    seq = repeat(seq, '... j -> ... i j', i=repeats.shape[-1])

    cumsum = rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = rearrange(cumsum_exclusive, '... i -> ... i 1')

    lengths = rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask


# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
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
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# compute pitch

def compute_pitch_pytorch(wav, sample_rate):
    # https://pytorch.org/audio/main/generated/torchaudio.functional.compute_kaldi_pitch.html#torchaudio.functional.compute_kaldi_pitch
    pitch_feature = torchaudio.functional.compute_kaldi_pitch(wav, sample_rate)
    pitch, nfcc = pitch_feature.unbind(dim=-1)
    return pitch


# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


# main class

class Attend(nn.Module):
    def __init__(
            self,
            dropout=0.,
            causal=False,
            use_flash=False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse(
            '2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=self.causal
            )

        return out

    def forward(self, q, k, v, mask=None, self_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        if self_mask is not None:
            self_mask = rearrange(self_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~self_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


class SpeechPromptEncoder(nn.Module):
    def __init__(
            self,
            dim_codebook,
            dims: Tuple[int] = (256, 2048, 2048, 2048, 2048, 512, 512, 512),
            *,
            depth=6,
            heads=8,
            dim_head=64,
            dropout=0.2,
            kernel_size=9,
            padding=4,
            use_flash_attn=True

    ):
        super().__init__()

        dims = [dim_codebook, *dims]

        self.dim, self.dim_out = dims[0], dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        modules = []
        for dim_in, dim_out in dim_pairs:
            modules.extend([
                nn.Conv1d(dim_in, dim_out, kernel_size, padding=padding),
                nn.SiLU()
            ])

        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            *modules,
            Rearrange('b c n -> b n c')
        )

        self.transformer = Transformer(
            dim=dims[-1],
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            use_flash=use_flash_attn
        )

    def forward(self, x):
        assert x.shape[-1] == self.dim

        x = self.conv(x)
        x = self.transformer(x)
        return x


# duration and pitch predictor seems to be the same

class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            kernel=3,
            groups=8,
            dropout=0.
    ):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            kernel,
            *,
            dropout=0.,
            groups=8,
            num_convs=2
    ):
        super().__init__()

        blocks = []
        for ind in range(num_convs):
            is_first = ind == 0
            dim_in = dim if is_first else dim_out
            block = Block(
                dim_in,
                dim_out,
                kernel,
                groups=groups,
                dropout=dropout
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        h = self.blocks(x)
        out = h + self.res_conv(x)
        return rearrange(out, 'b c n -> b n c')


def ConvBlock(dim, dim_out, kernel, dropout=0.):
    return nn.Sequential(
        Rearrange('b n c -> b c n'),
        nn.Conv1d(dim, dim_out, kernel, padding=kernel // 2),
        nn.SiLU(),
        nn.Dropout(dropout),
        Rearrange('b c n -> b n c'),
    )


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_context=None,
            num_latents=64,  # m in the paper
            dim_head=64,
            heads=8,
            ff_mult=4,
            use_flash_attn=False,
            dropout=0.1
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.embed_positions = (
            PositionalEmbedding(
                num_latents,
                dim,
                0,  # hard code to work
                learned=False,
            )
        )
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    use_flash=use_flash_attn,
                    cross_attn_include_queries=True,
                    dropout=dropout
                ),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]
        x = self.proj_context(x)  # B, Tx, D
        latents = repeat(self.latents, 'n d -> b n d', b=batch)  # B, #latent, D
        # add positional encoding for latents
        latent_mask = torch.ones(batch, latents.shape[1], device=x.device, dtype=torch.bool)
        latent_pos = self.embed_positions(latent_mask)
        latents = latents + latent_pos
        for attn, ff in self.layers:
            latents = attn(latents, context=x, self_mask=None, cross_mask=mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# model, which is wavenet + transformer

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.)
        return super().forward(causal_padded_x)


class WavenetResBlock(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dilation,
            kernel_size=3,
            skip_conv=False,
            dim_cond_mult=None
    ):
        super().__init__()

        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None

        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation=dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t=None):

        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, 'b c -> b c 1')
            t_gamma, t_beta = t.chunk(2, dim=-2)

        res = self.res_conv(x)

        x = self.conv(x)

        if self.cond:
            x = x * t_gamma + t_beta

        x = x.tanh() * x.sigmoid()

        x = x + res

        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)

        return x, skip


class WavenetStack(nn.Module):
    def __init__(
            self,
            dim,
            *,
            layers,
            kernel_size=3,
            has_skip=False,
            dim_cond_mult=None
    ):
        super().__init__()
        dilations = 2 ** torch.arange(layers)

        self.has_skip = has_skip
        self.blocks = mlist([])

        for dilation in dilations.tolist():
            block = WavenetResBlock(
                dim=dim,
                kernel_size=kernel_size,
                dilation=dilation,
                skip_conv=has_skip,
                dim_cond_mult=dim_cond_mult
            )

            self.blocks.append(block)

    def forward(self, x, t):
        residuals = []
        skips = []

        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)

        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)

            residuals.append(residual)
            skips.append(skip)

        if self.has_skip:
            return torch.stack(skips)

        return residuals


class Wavenet(nn.Module):
    def __init__(
            self,
            dim,
            *,
            stacks,
            layers,
            init_conv_kernel=3,
            dim_cond_mult=None
    ):
        super().__init__()
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        self.stacks = mlist([])

        for ind in range(stacks):
            is_last = ind == (stacks - 1)

            stack = WavenetStack(
                dim,
                layers=layers,
                dim_cond_mult=dim_cond_mult,
                has_skip=is_last
            )

            self.stacks.append(stack)

        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        for stack in self.stacks:
            x = stack(x, t)
        return self.final_conv(x.sum(dim=0))


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class ConditionableTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            ff_causal_conv=False,
            dim_cond_mult=None,
            cross_attn=False,
            use_flash=False,
            dropout_rate=0.1,
    ):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])
        cond = exists(dim_cond_mult)
        # if cond:
        #     self.adaLN_modulation = nn.Linear(dim, 6 * dim, bias=True)
        #     self.adaLN_modulation.weight.data.zero_()
        #     self.adaLN_modulation.bias.data.zero_()

        maybe_adaptive_norm_kwargs = dict(scale=not cond, dim_cond=dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        for _ in range(depth):
            self.layers.append(mlist([
                rmsnorm(dim),
                Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash, dropout=dropout_rate),
                rmsnorm(dim) if cross_attn else None,
                Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash,
                          dropout=dropout_rate) if cross_attn else None,
                rmsnorm(dim),
                FeedForward(dim=dim, mult=ff_mult, causal_conv=ff_causal_conv)
            ]))

        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim, bias=False)
        )

    def forward(
            self,
            x,
            times=None,
            context=None,
            self_mask=None,
            cross_mask=None,
    ):
        t = times
        # each of B x 1 x D shape
        # TODO: not used for now, maybe try later if the SEDD method is promising
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #     self.adaLN_modulation(t)[:, None].chunk(6, dim=2))
        # print("inside conditional transformer")

        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond=t)
            x = attn(x, self_mask=self_mask) + res

            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond=t)
                # here we attend to the latents (so there is no cross-attn mask required)
                # but we still need self mask
                x = cross_attn(x, context=context, self_mask=None, cross_mask=cross_mask) + res

            res = x
            x = ff_norm(x, cond=t)
            x = ff(x) + res

        return self.to_pred(x)


# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange('b d n -> b n d'),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        conv,
        nn.Linear(dim_inner, dim)
    )


# attention

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_context=None,
            causal=False,
            dim_head=64,
            heads=8,
            dropout=0.,
            use_flash=False,
            cross_attn_include_queries=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, self_mask=None, cross_mask=None):
        h, has_context = self.heads, exists(context)
        context = default(context, x)

        if has_context:
            if self.cross_attn_include_queries:
                context = torch.cat((x, context), dim=-2)
                # expand mask for the included query as well
                # this is only called for prompt, so append 1s to the front, no need to handle self-mask
                padded_mask = cross_mask.new_full((cross_mask.shape[0], x.shape[1]), True)
                cross_mask = torch.cat((padded_mask, cross_mask), dim=-1)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        out = self.attend(q, k, v, mask=cross_mask, self_mask=self_mask)  # B, #head, #latents, #head_dim
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer encoder

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            causal=False,
            dim_head=64,
            heads=8,
            use_flash=False,
            dropout=0.,
            ff_mult=4,
            final_norm=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                Attention(
                    dim,
                    causal=causal,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=dropout,
                    use_flash=use_flash
                ),
                RMSNorm(dim),
                FeedForward(
                    dim,
                    mult=ff_mult
                )
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, mask=None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask=mask) + x
            x = ff(ff_norm(x)) + x

        return self.norm(x)


# tensor helper functions

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def safe_div(numer, denom):
    return numer / denom.clamp(min=1e-10)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class TransformerWrapper(nn.Module):
    def __init__(self,
                 dim,
                 dim_cond_mult,
                 depth,
                 dim_head,
                 heads,
                 ff_mult,
                 num_latents_m=64,
                 ff_causal_conv=True,
                 use_flash=False,
                 cross_attn=True,
                 ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(
            512,
            dim,
            0,
            learned=False
        )
        # prompt condition
        self.null_prompt_tokens = nn.Parameter(torch.randn(num_latents_m, dim))
        nn.init.normal_(self.null_prompt_tokens, std=0.02)
        self.perceiver_resampler = PerceiverResampler(
            dim=dim,
            dim_context=768,
            num_latents=num_latents_m,
            depth=6,
            dim_head=dim_head,
            heads=heads,
            use_flash_attn=use_flash
        )

        self.transformer = ConditionableTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            ff_causal_conv=ff_causal_conv,
            dim_cond_mult=dim_cond_mult,
            use_flash=use_flash,
            cross_attn=cross_attn,
        )

    def forward(self, x, t, prompt, prompt_mask, prompt_cond_drop_mask, self_mask):
        resampled_prompt_tokens = self.perceiver_resampler(prompt, mask=prompt_mask)
        c = torch.where(
            rearrange(prompt_cond_drop_mask, 'b -> b 1 1'),
            self.null_prompt_tokens,
            resampled_prompt_tokens
        )
        pos_emb = self.pos_embed(self_mask)
        x += pos_emb
        x = self.transformer(x, t, context=c, self_mask=self_mask)
        return x, c


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        # x = self.norm_final(x)
        # x = self.dropout(x)
        x = self.linear(x)
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Define a linear layer.
        self.predictor = self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # Apply the linear layer
        x = self.predictor(x)
        # Apply the sigmoid activation function to get output in the range [0, 1]
        x = torch.sigmoid(x)
        return x


class ScoreModel(FairseqEncoder):
    def __init__(
            self,
            dim,
            vocab_size,
            graph_type="absorb",
            head_dim=64,
            condition_layers=12,
            ff_mult=4,
            noise_schedule='loglinear',
            dropout_rate=0.2
    ):
        super().__init__(None)
        self.scale_by_sigma = True
        self.dim = dim
        self.absorb = graph_type == "absorb"
        self.graph = Absorbing(vocab_size - 1) if self.absorb else Uniform(vocab_size)
        self.vocab_size = vocab_size
        self.noise_scheduler = LogLinearNoise() if noise_schedule == 'loglinear' else GeometricNoise()
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.sedd_vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.pos_embed = PositionalEmbedding(
            512,  # max len set to 512
            dim,
            0,  # hard code to work
            learned=False,
        )
        dim_time = 512
        dim_mult = dim_time // dim  # should be 2 if dim=512
        self.sigma_map = TimestepEmbedder(dim, frequency_embedding_size=256)  # 512 time embed

        n_heads = dim // head_dim
        self.sedd_transform = ConditionableTransformer(
            dim=dim,
            depth=condition_layers,
            dim_head=head_dim,
            heads=n_heads,
            ff_mult=ff_mult,
            ff_causal_conv=True,
            dim_cond_mult=dim_mult,
            use_flash=False,
            cross_attn=True,
            dropout_rate=dropout_rate
        )
        self.final_layer = DDitFinalLayer(dim, vocab_size, dim_time)


        self.predictor_transform = ConditionableTransformer(
            dim=dim,
            depth=6,
            dim_head=head_dim,
            heads=n_heads,
            ff_mult=ff_mult,
            ff_causal_conv=True,
            dim_cond_mult=None,
            use_flash=False,
            cross_attn=True,
            dropout_rate=dropout_rate
        )
        self.nat_predictor = BinaryClassifier(dim)
        self.nat_lm_head = nn.Linear(dim, vocab_size)
        self.bce_loss = nn.BCELoss(reduction="none")

    @staticmethod
    def sample_time(bz, device, sampling_eps=1e-3, ):
        return (1 - sampling_eps) * torch.rand(bz, device=device) + sampling_eps

    @staticmethod
    def get_acc_from_logits(logits, masked_indices, audio_units):
        most_probable_y = logits.argmax(dim=-1)
        correct = (audio_units[masked_indices] == most_probable_y[masked_indices]).sum()
        total = masked_indices.sum()
        acc = correct / total
        return acc

    def forward(
            self,
            audio_units,
            src_feature=None,
            src_mask=None,
            tgt_mask=None,
            nat_model=None,
            encoder_out=None,
            *args,
            **kwargs
    ):
        loss_dict = {}
        # ----------------- noise creation and injection ----------------- #
        tgt_lengths = tgt_mask.sum(dim=-1)
        time = self.sample_time(audio_units.shape[0], audio_units.device)
        # time[1] = 0.9 # debug purpose
        sigma, dsigma = self.noise_scheduler(time)
        # print(audio_units[0, :])
        able_to_noise_mask = (audio_units != 0) & (audio_units != 2)
        perturbed_batch = self.graph.sample_transition(audio_units, sigma[:, None])
        perturbed_batch[~able_to_noise_mask] = 0
        # add back eos token
        perturbed_batch = torch.scatter(perturbed_batch, -1, tgt_lengths[:, None] - 1, 2)

        masked_indices = perturbed_batch == 1004
        pos_embed = self.pos_embed(tgt_mask)
        # ---------------- Pass through score model to estimate ratio for refinement---------------- #
        sedd_embed = self.sedd_vocab_embed(perturbed_batch)
        sedd_x = sedd_embed + pos_embed
        cond = F.silu(self.sigma_map(sigma))  # B x dim
        sedd_x = self.sedd_transform(sedd_x, cond, context=src_feature, self_mask=tgt_mask, cross_mask=src_mask)
        sedd_x = self.final_layer(sedd_x, cond)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(sedd_x.dtype)[:, None,
                         None]
            sedd_x = sedd_x - esigm1_log - np.log(sedd_x.shape[-1] - 1)  # this will be approximately averaged at 0

            # self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
            # indices[..., None]: BxTx1, same as the zero vectors
            # then we make score[i][j][corresponding_token] = 0 as ith token itself cannot be the next token
        log_score = torch.scatter(sedd_x, -1, perturbed_batch[..., None], torch.zeros_like(sedd_x[..., :1]))
        acc = self.get_acc_from_logits(log_score, masked_indices, audio_units)
        loss_dict["score_acc"] = acc
        # ------------------ Finish Score Estimation ---------------------#
        loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, audio_units)
        weighted_loss = (dsigma[:, None] * loss)
        weighted_loss[~able_to_noise_mask] = 0
        score_loss = weighted_loss.sum(dim=-1)
        score_loss = score_loss.mean()
        loss_dict["score_loss"] = score_loss

        # ------------------ Use pre-trained NAT model to learn Predictor ---------------------#
        tokens_for_nat = perturbed_batch.clone()
        tokens_for_nat[masked_indices] = nat_model.unk
        tokens_for_nat[~tgt_mask] = nat_model.pad

        with torch.no_grad():
            nat_pred, _ = nat_model.decoder(
                tokens_for_nat,
                encoder_out=encoder_out,
                normalize=False,
            )

        nat_token = nat_pred.argmax(dim=-1)
        tokens_for_nat[masked_indices] = nat_token[masked_indices]
        correct_tokens = tokens_for_nat[masked_indices] == audio_units[masked_indices]
        nat_acc = correct_tokens.sum() / able_to_noise_mask.sum()
        loss_dict["nat_acc"] = nat_acc

        token_embed = self.vocab_embed(tokens_for_nat)
        x = token_embed + pos_embed
        # x-attn to src features
        x = self.predictor_transform(x, context=src_feature, self_mask=tgt_mask, cross_mask=src_mask)
        binary_pred = self.nat_predictor(x).squeeze(-1)
        binary_groundtruth = correct_tokens.float()
        # positive instances is normally larger so need to balance the weight
        num_pos = binary_groundtruth.sum()
        num_neg = (1-binary_groundtruth).sum()
        weight = binary_groundtruth * (num_neg / num_pos) + (1 - binary_groundtruth) * (num_pos / num_neg)
        bce_loss = self.bce_loss(binary_pred[masked_indices], binary_groundtruth)
        bce_loss = (bce_loss * weight).mean()
        loss_dict["bce_loss"] = bce_loss
        binary_pred_acc = (binary_pred > 0.5).float()[masked_indices] == correct_tokens
        binary_pred_acc = binary_pred_acc.sum() / masked_indices.sum()
        loss_dict["bce_acc"] = binary_pred_acc
        return loss_dict


    def sedd_forward(self, tokens, sigma, tgt_mask=None, src_feature=None, src_mask=None):
        pos_embed = self.pos_embed(tgt_mask)
        sedd_embed = self.sedd_vocab_embed(tokens)
        sedd_x = sedd_embed + pos_embed
        cond = F.silu(self.sigma_map(sigma))  # B x dim
        sedd_x = self.sedd_transform(sedd_x, cond, context=src_feature, self_mask=tgt_mask, cross_mask=src_mask)
        sedd_x = self.final_layer(sedd_x, cond)
        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(sedd_x.dtype)[:, None,
                         None]
            sedd_x = sedd_x - esigm1_log - np.log(sedd_x.shape[-1] - 1)  # this will be
        log_score = torch.scatter(sedd_x, -1, tokens[..., None], torch.zeros_like(sedd_x[..., :1]))
        return log_score



    def nat_forward(self, tokens, src_feature=None, src_mask=None, tgt_mask=None):
        token_embed = self.vocab_embed(tokens)
        pos_embed = self.pos_embed(tgt_mask)
        x = token_embed + pos_embed
        # x-attn to src features
        x = self.nat_transform(x, context=src_feature, self_mask=tgt_mask, cross_mask=src_mask)
        # nll loss for unmasking
        nat_lm_head = self.nat_lm_head(x)
        return nat_lm_head


    def predictor_forward(self, tokens, src_feature=None, src_mask=None, tgt_mask=None):
        token_embed = self.vocab_embed(tokens)
        pos_embed = self.pos_embed(tgt_mask)
        x = token_embed + pos_embed
        # x-attn to src features
        x = self.predictor_transform(x, context=src_feature, self_mask=tgt_mask, cross_mask=src_mask)
        x = self.nat_predictor(x).squeeze(-1)
        return x


    def test_forward(
            self,
            perturbed_batch,
            sigma,
            src_feature=None,
            src_mask=None,
            tgt_mask=None,
    ):
        # ----------------- noise creation and injection performed in outer loop ----------------- #
        sigma = sigma.reshape(-1)
        cond = F.silu(self.sigma_map(sigma))  # B x dim

        # ---------------- Pass through score model to estimate ratio ---------------- #
        token_embed = self.vocab_embed(perturbed_batch)
        pos_embed = self.pos_embed(tgt_mask)
        x = token_embed + pos_embed
        # learn src feature with latents, Bx #latents=64 x D
        prompt_tokens = self.perceiver_resampler(src_feature, mask=src_mask)
        # x-attn to src features
        x = self.condition_transform(x, cond, context=prompt_tokens, self_mask=tgt_mask)
        save_x = x.clone()
        nat_feature = self.nat_layers(save_x, cond, context=prompt_tokens, self_mask=tgt_mask)
        nat_lm_head = self.nat_lm_head(nat_feature)

        x = self.final_layer(x, cond)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            # shape = 1, 1, 1
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)  # this will be approximately averaged at 0

            # self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
            # indices[..., None]: BxTx1, same as the zero vectors
            # then we make score[i][j][corresponding_token] = 0 as ith token itself cannot be the next token
        log_score = torch.scatter(x, -1, perturbed_batch[..., None], torch.zeros_like(x[..., :1]))
        return log_score, nat_lm_head

    def nat_inference(self, batch_dims, steps, eps=1e-5, device='cuda',
                      src_feature=None,
                      src_mask=None,
                      tgt_mask=None,
                      target_units=None,
                      nar_units=None):
        nar_str = self.unit_to_str(nar_units)
        ref_str = self.unit_to_str(target_units)
        nar_bleu = sacrebleu.sentence_bleu(nar_str, [ref_str]).score
        print(nar_bleu)
        x = self.graph.sample_limit(*batch_dims).to(device)
        batch_size, num_tokens = x.shape
        output_tokens = x
        output_scores = x.new_full(x.shape, 0.0).float()
        for i in range(steps):
            output_masks = output_tokens == 1004
            nat_lm_head = self.nat_forward(output_tokens, src_feature=src_feature, src_mask=src_mask, tgt_mask=tgt_mask)
            _scores, _tokens = torch.max(nat_lm_head, dim=-1)
            # print(_scores)
            # _scores = self.predictor_forward(output_tokens, src_feature=src_feature, src_mask=src_mask, tgt_mask=tgt_mask)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            skeptical_mask = _skeptical_unmasking(  # p%分数最低的 加入mask
                output_scores, output_tokens.ne(0), 1 - (i + 1) / steps
            )
            if i < steps - 1:
                output_tokens.masked_fill_(skeptical_mask, 1004)
                output_scores.masked_fill_(skeptical_mask, 0.0)

        nat_str = self.unit_to_str(output_tokens - 4)
        bleu = sacrebleu.sentence_bleu(nat_str, [ref_str]).score
        print(bleu)
        return output_tokens


    def inference(self, steps, batch_dims, proj_fun=lambda x: x, eps=1e-5, device='cuda',
                  src_feature=None, src_mask=None, tgt_mask=None, denoise=True, target_units=None, nar_units=None):
        predictor = AnalyticPredictor(self.graph, self.noise_scheduler)
        projector = proj_fun
        denoiser = Denoiser(self.graph, self.noise_scheduler)
        x = self.graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        ref_str = self.unit_to_str(target_units)
        nar_str = self.unit_to_str(nar_units)
        nar_bleu = sacrebleu.sentence_bleu(nar_str, [ref_str]).score

        # nat_result = self.nat_inference(x, 50, eps=eps, src_feature=src_feature, src_mask=src_mask, tgt_mask=tgt_mask, ref_str=ref_str)

        # print(nar_bleu)
        # print(x)
        # print(timesteps)
        best_score = -1
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            ################# Euler Predictor #####################
            # sigma, dsigma = self.noise_scheduler(t)
            # score = self.test_forward(x, sigma,
            #                   src_feature=src_feature,
            #                   src_mask=src_mask,
            #                   tgt_mask=tgt_mask)
            # rev_rate = dt * dsigma[..., None] * self.graph.reverse_rate(x, score)
            # x = self.graph.sample_rate(x, rev_rate)

            ################ Analytic Predictor #####################
            curr_sigma = self.noise_scheduler(t)[0]
            next_sigma = self.noise_scheduler(t - dt)[0]
            # print(curr_sigma, next_sigma)
            dsigma = curr_sigma - next_sigma
            # ------------ TEST Time Score Computation ------------ #
            score = self.test_forward(x, curr_sigma,
                                      src_feature=src_feature,
                                      src_mask=src_mask,
                                      tgt_mask=tgt_mask)
            score = score.exp()
            # print(score[0, :])
            stag_score = self.graph.staggered_score(score, dsigma)
            probs = stag_score * self.graph.transp_transition(x, dsigma)
            if self.graph.absorb and i == steps - 1:
                probs = probs[..., :-1]
            x = sample_categorical(probs)

            pred_tokens = self.unit_to_str(x - 4)
            bleu_score = sacrebleu.sentence_bleu(pred_tokens, [ref_str]).score

            if bleu_score > best_score:
                best_score = bleu_score
                best_tokens = x
        print(f"ar result:{nar_bleu}, ssed result: {best_score}")
        return best_tokens


    def sedd_refine(self, input_tokens, steps, eps=1e-5, device='cuda', tgt_mask=None, src_feature=None, src_mask=None):
        " input_tokens: B x T generated from NAT, has some masked tokens"
        input_tokens[input_tokens == 3] = 1004
        masked_indices = input_tokens == 1004

        num_masked = (input_tokens == 1004).sum(dim=1)
        num_all = tgt_mask.sum(dim=1)
        batch_size = input_tokens.shape[0]

        # (1 - (1- (-sigma).exp) ) * num_tokens = num_tokens_to_decode
        sigma = -torch.log(1 - (num_masked / num_all))
        cur_t = ((-sigma).exp() - 1) / (1e-3 - 1)
        cur_t = cur_t.squeeze()
        all_timesteps =[]
        for batch_id in range(batch_size):
            start_t = cur_t[batch_id]
            timesteps = torch.linspace(start_t, eps, steps + 1, device=device)
            all_timesteps.append(timesteps)
        timesteps = torch.stack(all_timesteps)
        dt = (1 - eps) / steps
        x = input_tokens
        for i in range(steps):
            t = timesteps[:, i]
            curr_sigma = self.noise_scheduler(t)[0]
            next_sigma = self.noise_scheduler(t - dt)[0]

            ################ Analytic Predictor #####################
            dsigma = curr_sigma - next_sigma
            # ------------ TEST Time Score Computation ------------ #
            # print(x.shape, curr_sigma.shape)
            # curr_sigma = curr_sigma.squeeze(1)
            score = self.sedd_forward(x, curr_sigma,
                                      tgt_mask=tgt_mask,
                                      src_feature=src_feature,
                                      src_mask=src_mask)
            score = score.exp()

            stag_score = self.graph.staggered_score(score, dsigma[:, None])
            probs = stag_score * self.graph.transp_transition(x, dsigma[:, None])
            if self.graph.absorb and i == steps - 1:
                probs = probs[..., :-1]
            x_bak = x.clone()
            x = sample_categorical(probs)
            # print(x[0, :])
            x[~masked_indices] = x_bak[~masked_indices]
        return x






    def curriculum_inference(self, steps, batch_dims, proj_fun=lambda x: x, eps=1e-5, device='cuda',
                             src_feature=None, src_mask=None, tgt_mask=None, denoise=True, target_units=None,
                             nar_units=None, nat_steps=None):
        projector = proj_fun
        x = self.graph.sample_limit(*batch_dims).to(device)

        ref_str = self.unit_to_str(target_units)
        nar_str = self.unit_to_str(nar_units)
        nar_bleu = sacrebleu.sentence_bleu(nar_str, [ref_str]).score
        best_score = -1
        batch_size, num_tokens = x.shape
        counter = 0
        fixed_conditional_indices = None
        decode_with_nat = True

        # we first decode with nat for a few steps then refine with sedd
        # we only use nat to decode (nat_ratio)% of token, and decode the rest with diffusion
        ratio = 0.8
        # nat_steps = int(ratio * 20)
        decode_token_upperbound = torch.tensor(num_tokens * ratio).to(device)
        for i in range(nat_steps):
            # TODO: each decode handles the same amount of unmasking, maybe make this scheduling better
            num_tokens_to_decode = (i + 1) * decode_token_upperbound / nat_steps
            # (1 - (1- (-sigma).exp) ) * num_tokens = num_tokens_to_decode
            sigma = -torch.log(num_tokens_to_decode / num_tokens)
            _, nat_head = self.test_forward(x, sigma,
                                            src_feature=src_feature,
                                            src_mask=src_mask,
                                            tgt_mask=tgt_mask)
            nat_head = F.log_softmax(nat_head, dim=-1)
            _score, _tokens = torch.max(nat_head, dim=-1)
            topk_score, topk_indices = torch.topk(_score, k=int(torch.ceil(num_tokens_to_decode).item()))
            x = x.new_full(x.shape, 1004)
            range_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, topk_indices.size(1))
            x[range_idx, topk_indices] = _tokens[range_idx, topk_indices]
        fixed_conditional_indices = topk_indices
        # reverse current sigma back to time t
        # print("sigma after nat: ", sigma.item())
        cur_t = ((-sigma).exp() - 1) / (1e-3 - 1)
        # curr_sigma = self.noise_scheduler(cur_t)[0]

        timesteps = torch.linspace(cur_t, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            curr_sigma = self.noise_scheduler(t)[0]
            next_sigma = self.noise_scheduler(t - dt)[0]
            ################ Analytic Predictor #####################
            dsigma = curr_sigma - next_sigma
            # ------------ TEST Time Score Computation ------------ #
            score, _ = self.test_forward(x, curr_sigma,
                                         src_feature=src_feature,
                                         src_mask=src_mask,
                                         tgt_mask=tgt_mask)
            score = score.exp()
            # print(score[0, :])
            stag_score = self.graph.staggered_score(score, dsigma)
            probs = stag_score * self.graph.transp_transition(x, dsigma)
            if self.graph.absorb and i == steps - 1:
                probs = probs[..., :-1]
            x_bak = x.clone()
            x = sample_categorical(probs)
            if fixed_conditional_indices is not None:
                range_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, fixed_conditional_indices.size(1))
                x[range_idx, fixed_conditional_indices] = x_bak[range_idx, fixed_conditional_indices]
            pred_tokens = self.unit_to_str(x - 4)
            bleu_score = sacrebleu.sentence_bleu(pred_tokens, [ref_str]).score
            # if bleu_score > best_score:
            #     best_score = bleu_score
            #     best_tokens = x
            counter += 1
            # print(bleu_score)
        print(f"ar result:{nar_bleu}, ssed result: {bleu_score}")
        return x, nar_str, pred_tokens, ref_str

    @property
    def device(self):
        return next(self.model.parameters()).device

    def print(self, s):
        return self.accelerator.print(s)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def unit_to_str(self, units):
        np_unit = units.squeeze().cpu().numpy()
        out = [str(np_unit[i]) for i in range(np_unit.shape[0])]
        out_str = " ".join(out)
        return out_str

    def inject_noise(self, audio_token, audio_mask, times, unk_token=None):
        """
        Injects masks into the input sequence x based on the timestep t.
        audio_token: B, Ty
        times: B, 1
        """
        sigma_t = torch.sin(math.pi * times / (2 * self.timesteps))
        # draw mask based on the Bernoulli distribution with probability sigma_t
        M = sigma_t.unsqueeze(1).expand(-1, audio_token.shape[1])
        M = torch.bernoulli(M)
        M = M.masked_fill_(~audio_mask, 0).bool()
        masked_token = audio_token.masked_fill_(M, unk_token)
        return masked_token
