import math
import copy
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from fairseq.modules import PositionalEmbedding
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
from tqdm.auto import tqdm

# constants

mlist = nn.ModuleList


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


# compute pitch

def compute_pitch_pytorch(wav, sample_rate):
    # https://pytorch.org/audio/main/generated/torchaudio.functional.compute_kaldi_pitch.html#torchaudio.functional.compute_kaldi_pitch
    pitch_feature = torchaudio.functional.compute_kaldi_pitch(wav, sample_rate)
    pitch, nfcc = pitch_feature.unbind(dim=-1)
    return pitch


# as mentioned in paper using pyworld

def compute_pitch_pyworld(wav, sample_rate, hop_length, pitch_fmax=640.0):
    is_tensor_input = torch.is_tensor(wav)

    if is_tensor_input:
        device = wav.device
        wav = wav.contiguous().cpu().numpy()

    if divisible_by(len(wav), hop_length):
        wav = np.pad(wav, (0, hop_length // 2), mode="reflect")

    wav = wav.astype(np.double)

    outs = []

    for sample in wav:
        f0, t = pw.dio(
            sample,
            fs=sample_rate,
            f0_ceil=pitch_fmax,
            frame_period=1000 * hop_length / sample_rate,
        )

        f0 = pw.stonemask(sample, f0, t, sample_rate)
        outs.append(f0)

    outs = np.stack(outs)

    if is_tensor_input:
        outs = torch.from_numpy(outs).to(device)

    return outs


def f0_to_coarse(f0, f0_bin=256, f0_max=1100.0, f0_min=50.0):
    f0_mel_max = 1127 * torch.log(1 + torch.tensor(f0_max) / 700)
    f0_mel_min = 1127 * torch.log(1 + torch.tensor(f0_min) / 700)

    f0_mel = 1127 * (1 + f0 / 700).log()
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).int()
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


# peripheral models

# audio to mel

from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


# helpers

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
            use_flash_attn=False
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
                    dropout=0.1
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
            use_flash=False
    ):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])

        cond = exists(dim_cond_mult)

        maybe_adaptive_norm_kwargs = dict(scale=not cond, dim_cond=dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        for _ in range(depth):
            self.layers.append(mlist([
                rmsnorm(dim),
                Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash, dropout=0.1),
                rmsnorm(dim) if cross_attn else None,
                Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash,
                          dropout=0.1) if cross_attn else None,
                rmsnorm(dim),
                FeedForward(dim=dim, mult=ff_mult, causal_conv=ff_causal_conv)
            ]))

        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias=False)
        )

    def forward(
            self,
            x,
            times=None,
            context=None,
            self_mask=None,
    ):
        t = times
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
                x = cross_attn(x, context=context, self_mask=None) + res

            res = x
            x = ff_norm(x, cond=t)
            x = ff(x) + res

        return self.to_pred(x)


class Model(nn.Module):
    def __init__(
            self,
            dim,
            latent_dim,
            *,
            depth=12,
            dim_head=64,
            heads=8,
            ff_mult=4,
            wavenet_layers=8,
            wavenet_stacks=4,
            dim_cond_mult=4,
            use_flash_attn=False,
            dim_prompt=None,
            num_latents_m=64,  # number of latents to be perceiver resampled ('q-k-v' with 'm' queries in the paper)
            resampler_depth=2,
            cond_drop_prob=0.,
            condition_on_prompt=False
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        # first transform dim to hid dim as the latent dim is much smaller
        self.init_conv = nn.Conv1d(latent_dim, dim, 1)



        # time condition
        dim_time = dim * dim_cond_mult

        self.to_time_cond = Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim + 1, dim_time),
            nn.SiLU()
        )

        # prompt condition
        self.cond_drop_prob = cond_drop_prob  # for classifier free guidance
        self.condition_on_prompt = condition_on_prompt
        self.to_prompt_cond = None

        if self.condition_on_prompt:
            self.null_prompt_cond = nn.Parameter(torch.randn(dim_time))
            self.null_prompt_tokens = nn.Parameter(torch.randn(num_latents_m, dim))

            nn.init.normal_(self.null_prompt_cond, std=0.02)
            nn.init.normal_(self.null_prompt_tokens, std=0.02)

            self.to_prompt_cond = Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_prompt, dim_time),
                nn.SiLU()
            )

            self.perceiver_resampler = PerceiverResampler(
                dim=dim,
                dim_context=dim_prompt,
                num_latents=num_latents_m,
                depth=resampler_depth,
                dim_head=dim_head,
                heads=heads,
                use_flash_attn=use_flash_attn
            )
        self.pos_embed = PositionalEmbedding(
            1024,
            dim,
            0,
            learned=False
        )

        # aligned conditioning from aligner + duration module
        self.null_cond = None
        self.cond_to_model_dim = None
        dim_cond_mult = dim_cond_mult * (2 if condition_on_prompt else 1)

        # wavenet
        self.wavenet = Wavenet(
            dim=dim,
            stacks=wavenet_stacks,
            layers=wavenet_layers,
            dim_cond_mult=dim_cond_mult
        )

        # transformer
        self.transformer = ConditionableTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            ff_causal_conv=True,
            dim_cond_mult=dim_cond_mult,
            use_flash=use_flash_attn,
            cross_attn=condition_on_prompt,
        )

        self.final_proj = nn.Linear(dim, latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            x,
            times,
            prompt=None,
            prompt_mask=None,
            input_mask=None,
            cond=None,
            cond_drop_prob=None
    ):
        b = x.shape[0]
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        t = self.to_time_cond(times)  # B x time_dim
        c = None

        if exists(self.to_prompt_cond):
            prompt_cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)
            masked_prompt = prompt.masked_fill_(~prompt_mask.unsqueeze(2), 0)
            prompt_cond = self.to_prompt_cond(masked_prompt)
            prompt_cond = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1'),
                self.null_prompt_cond,
                prompt_cond,
            )
            # B x 2D (first half is for time, second half is for prompt's mean pooled feature)
            t = torch.cat((t, prompt_cond), dim=-1)
            resampled_prompt_tokens = self.perceiver_resampler(prompt, mask=prompt_mask)
            c = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1 1'),
                self.null_prompt_tokens,
                resampled_prompt_tokens
            )

        # rearrange to channel first

        x = rearrange(x, 'b n d -> b d n')
        x = self.init_conv(x)
        x = self.wavenet(x, t)
        x = rearrange(x, 'b d n -> b n d')
        pos_emb = self.pos_embed(input_mask)
        x += pos_emb

        if c is not None:
            # print(c.shape)
            x = self.transformer(x, t, context=c, self_mask=input_mask)
        else:
            x = self.transformer(x, t, self_mask=input_mask)
        x = self.final_proj(x)
        return x


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

        if has_context and self.cross_attn_include_queries:
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


# noise schedules

def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.)


# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale=1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)


def gamma_to_log_snr(gamma, scale=1, eps=1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps=eps)




class WavenetEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            stacks,
            layers,
            init_conv_kernel=3,
            dim_cond_mult=None
    ):
        super().__init__()
        self.init_conv = CausalConv1d(input_dim, output_dim, init_conv_kernel)
        self.stacks = mlist([])

        for ind in range(stacks):
            is_last = ind == (stacks - 1)
            stack = WavenetStack(
                output_dim,
                layers=layers,
                dim_cond_mult=dim_cond_mult,
                has_skip=is_last
            )
            self.stacks.append(stack)
        self.final_conv = CausalConv1d(output_dim, output_dim, 1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        for stack in self.stacks:
            x = stack(x, t)
        return self.final_conv(x.sum(dim=0))


class SpeechVAEEncoderDecoder(FairseqEncoder):
    def __init__(self,
                 dim=768,
                 latent_dim=16,
                 ):
        super().__init__(None)
        self.dim = dim
        self.latent_dim = latent_dim
        # downsample the input dim from 768 -> 32, and split into two chunks, one for gaussian mean prediction and another for variance prediction
        if latent_dim == 16:
            # 768 -> 192 -> 64 -> 32
            chan_mults = [4, 3, 2]
        elif latent_dim == 32:
            # 768 -> 192
            chan_mults = [4, 3]
        elif latent_dim == 128:
            chan_mults = [3]

        cur_dim = dim
        encoder_blocks = nn.ModuleList()
        decoder_blocks = nn.ModuleList()
        for mult in chan_mults:
            target_dim = cur_dim // mult
            wavenet_block = WavenetEncoder(
                input_dim=cur_dim,
                output_dim=target_dim,
                stacks=2,
                layers=3,
            )
            encoder_blocks.append(wavenet_block)
            cur_dim = target_dim

        first_step = True
        for mult in reversed(chan_mults):
            target_dim = cur_dim * mult
            if first_step:
                cur_dim = cur_dim // 2  # after gaussian smaple, the first starting dim is reduced by 2
                first_step = False
            # print(cur_dim, target_dim)
            wavenet_block = WavenetEncoder(
                input_dim=cur_dim,
                output_dim=target_dim,
                stacks=2,
                layers=3,
            )
            decoder_blocks.append(wavenet_block)
            cur_dim = target_dim
        self.encoder_wave = encoder_blocks
        self.decoder_wave = decoder_blocks
        self.decoder_tf = ConditionableTransformer(
            dim=dim,
            depth=6,
            dim_head=96,
            heads=8,
            ff_mult=4,
            ff_causal_conv=True,
            dim_cond_mult=None,
            use_flash=False,
            cross_attn=False,
        )
        vocab_size = 1004  # hard code the size of dictionary
        self.decoder_lm = nn.Linear(dim, vocab_size)
        self.mse_loss = nn.MSELoss()

    @torch.no_grad()
    def encode_feature(self, feature):
        " feature: B, T, D"
        x = feature.transpose(1, 2)
        for encoder_wavenet in self.encoder_wave:
            x = encoder_wavenet(x)
        # convert into gaussian sample
        posterior = DiagonalGaussianDistribution(x)
        return posterior.sample()

    def decode_feature(self, latent, mask):
        " latent: B, T, D, mask: B, T"
        x = latent.transpose(1, 2)
        for decoder_wavenet in self.decoder_wave:
            x = decoder_wavenet(x)
        decoded_feature = self.decoder_tf(x.transpose(1, 2), times=None, context=None, self_mask=mask)
        lm_result = self.decoder_lm(decoded_feature)
        return decoded_feature, lm_result

    def forward(self, input_feature, input_token, mask):
        # encoding step
        x = input_feature.transpose(1, 2)
        for encoder_wavenet in self.encoder_wave:
            x = encoder_wavenet(x)
        # convert into gaussian sample
        posterior = DiagonalGaussianDistribution(x)
        x = posterior.sample()
        kl_loss = posterior.kl_3d(mask=mask)
        kl_loss = kl_loss.mean()

        # decode back to original dim
        for decoder_wavenet in self.decoder_wave:
            x = decoder_wavenet(x)

        decoded_feature = self.decoder_tf(x.transpose(1, 2), times=None, context=None, self_mask=mask)

        feature_mask = mask.unsqueeze(2).expand(-1, -1, decoded_feature.shape[2])
        selected_pred = decoded_feature[feature_mask]
        selected_true = input_feature[feature_mask]
        mse_loss = self.mse_loss(selected_pred, selected_true)

        # cross entropy loss
        lm_result = self.decoder_lm(decoded_feature)
        return mse_loss, lm_result, kl_loss


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


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


class DDPMScheduler:
    def __init__(self, timesteps, scale=1.0):
        self.num_timesteps = timesteps
        self.scale = scale  # should be 1 during training, but can be adjusted for faster sampling during inference
        betas = get_named_beta_schedule("cosine", timesteps)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)

        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )


    def get_beta(self, t, shape):
        return _extract_into_tensor(self.betas, t, shape)

    def get_snr(self, t):
        sqrt_alpha_cum = self.get_sqrt_alpha_cum(t, t.shape)
        sqrt_one_minus_alpha_cum = self.get_sqrt_one_minus_alpha_cum(t, t.shape)
        return (sqrt_alpha_cum ** 2) / (sqrt_one_minus_alpha_cum ** 2)

    def get_sqrt_alpha_cum(self, t, shape):
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, shape)

    def get_alpha_cum(self, t, shape):
        return _extract_into_tensor(self.alphas_cumprod, t, shape)

    def get_alpha_prev_cum(self, t, shape):
        return _extract_into_tensor(self.alphas_cumprod_prev, t, shape)

    def get_sqrt_one_minus_alpha_cum(self, t, shape):
        return _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, shape)


class LatentDiscreteModel(FairseqEncoder):
    def __init__(
            self,
            speech_decoder,
            dim, # latent dimension processed from argument
            latent_dim,
            target_sample_hz=None,
            timesteps=1000,
            use_ddim=True,
            noise_schedule='sigmoid',
            objective='v',
            schedule_kwargs: dict = dict(),
            time_difference=0.,
            min_snr_loss_weight=True,
            min_snr_gamma=5,
            train_prob_self_cond=0.9,
            scale=1.,  # this will be set to < 1. for better convergence when
            use_cond=False,
            multitask=True,
    ):
        super().__init__(None)
        self.speech_decoder = speech_decoder.encoder
        # latent is 128D, we first multiply it to 4x128=512D
        self.use_cond = use_cond
        self.multitask = multitask

        # dim is small, we first need to map it to higher dim (like 512)
        if use_cond:
            self.model = Model(
                dim,
                latent_dim,
                condition_on_prompt=True,
                dim_prompt=768,
                num_latents_m=64
            )
        else:
            self.model = Model(
                dim,
                latent_dim,
                condition_on_prompt=False
            )

        self.conditional = False  # always true as source feature is given
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = None

        # ---------------------------- Setup DDPM ---------------------------- #
        self.scheduler = DDPMScheduler(timesteps, scale=scale)
        self.dim = dim
        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training
        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400
        self.time_difference = time_difference
        self.train_prob_self_cond = train_prob_self_cond

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.mse_loss = nn.MSELoss("mean")

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

    @torch.no_grad()
    def ddim_sample(self,
                    tgt_feature,
                    prompt=None,
                    prompt_mask=None,
                    input_mask=None,
                    cond_scale=1.,
                    ref_units=None,
                    start_step=50,
                    ):

        self.speech_decoder.eval()
        encode_tgt_feature = self.speech_decoder.encode_feature(tgt_feature).transpose(1, 2)
        shape = encode_tgt_feature.shape
        batch, device = shape[0], self.device
        # backward diffusion from the middle
        start_time = start_step
        timesteps = list(range(start_time))[::-1]

        # introduce synthetic noise to the encoded feature
        t_start = encode_tgt_feature.new_full((batch,), start_time).long()
        sqrt_alpha_cum = self.scheduler.get_sqrt_alpha_cum(t_start, shape)
        sqrt_one_minus_alpha_cum = self.scheduler.get_sqrt_one_minus_alpha_cum(t_start, shape)
        # print(sqrt_alpha_cum[0, 0, 0], sqrt_one_minus_alpha_cum[0, 0, 0])
        x = sqrt_alpha_cum * encode_tgt_feature + sqrt_one_minus_alpha_cum * torch.randn(shape, device=device)

        for step_id, time in enumerate(timesteps):
            t = encode_tgt_feature.new_full((batch,), time).long()
            noise = self.model(
                x, t,
                input_mask=input_mask,  # mask for self-attn
                cond_drop_prob=0
            )

            sqrt_alpha_cum = self.scheduler.get_sqrt_alpha_cum(t, noise.shape)
            sqrt_one_minus_alpha_cum = self.scheduler.get_sqrt_one_minus_alpha_cum(t, noise.shape)

            x_1_hat = safe_div(x - sqrt_one_minus_alpha_cum * noise, sqrt_alpha_cum)
            # tilde_epsilon = (x - \sqrt(alpha_t) x_start) / \sqrt(1 - alpha_t)
            pred_noise = safe_div(x - sqrt_alpha_cum * x_1_hat, sqrt_one_minus_alpha_cum)

            alpha_bar = self.scheduler.get_alpha_cum(t, noise.shape)
            alpha_bar_prev = self.scheduler.get_alpha_prev_cum(t, noise.shape)
            eta = 0
            sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise = torch.randn_like(x)
            mean_pred = (x_1_hat * torch.sqrt(alpha_bar_prev)
                         + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
                         )
            if time > 1:
                x = mean_pred + sigma * noise
            else:
                x = mean_pred

            if time == 1:
                break

        # --------- decode back to hubert dim, and go through LM head ---------------#
        recon_feature, pred_lm_heads = self.speech_decoder.decode_feature(x, input_mask)
        # print(x_0.shape, pred_lm_heads.shape)
        pred_units = torch.argmax(pred_lm_heads, dim=-1)
        pred_units = pred_units - 4
        # compute accuracy
        match = (pred_units[input_mask] == ref_units[input_mask]).sum().item()
        total = input_mask.sum().item()
        acc = match / total
        # print(f"accuracy: {acc}")
        # compute bleu score btw generated and reference units
        tgt_len = input_mask.sum(dim=1)
        refs, hyps= [], []
        out_tokens = []
        for i in range(batch):
            # pred_str = self.unit_to_str(pred_units[i][:tgt_len[i]])
            # ref_str = self.unit_to_str(ref_units[i][:tgt_len[i]])
            # refs.append(ref_str)
            # hyps.append(pred_str)
            out_tokens.append(pred_units[i][:tgt_len[i]])
            # print(f"pred: {pred_str} ref: {ref_str}")

        # bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score
        # print(bleu_score)
        return out_tokens, match, total, recon_feature

    def process_prompt(self, prompt=None):
        if not exists(prompt):
            return None

        assert self.model.condition_on_prompt

        is_raw_prompt = prompt.ndim == 2
        assert not (is_raw_prompt and not exists(
            self.codec)), 'codec must be passed in if one were to train on raw prompt'

        if is_raw_prompt:
            with torch.no_grad():
                self.codec.eval()
                prompt, _, _ = self.codec(prompt, curtail_from_left=True, return_encoded=True)

        return prompt

    def expand_encodings(self, phoneme_enc, attn, pitch):
        expanded_dur = einsum('k l m n, k j m -> k j n', attn, phoneme_enc)
        pitch_emb = self.pitch_emb(rearrange(f0_to_coarse(pitch), 'b 1 t -> b t'))
        pitch_emb = rearrange(pitch_emb, 'b t d -> b d t')
        expanded_pitch = einsum('k l m n, k j m -> k j n', attn, pitch_emb)
        expanded_encodings = expanded_dur + expanded_pitch
        return expanded_encodings

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



    def forward(
            self,
            audio,
            audio_units,
            src_feature=None,
            src_mask=None,
            tgt_mask=None,
            prompt=None,
            pitch=None,
            *args,
            **kwargs
    ):

        # t from Unif(1,T) instead of (0,T) to avoid x_0
        times = torch.randint(1, self.timesteps, (audio.shape[0],), device=self.device)
        # ---------- First encode the feature into smaller latent -------------#
        self.speech_decoder.eval()
        # B x T x z=16 (16 dimensional guassian latent) or z= 128, depends on the vae used
        audio_feature = self.speech_decoder.encode_feature(audio).transpose(1, 2)
        # ---------- x_1 = Enc(x_0) + beta_0 * epsilon ------------------------ #
        t_0 = torch.zeros_like(times)
        beta_0 = self.scheduler.get_beta(t_0, audio_feature.shape)
        x_1 = audio_feature + torch.randn_like(audio_feature) * beta_0

        sqrt_alpha_cum = self.scheduler.get_sqrt_alpha_cum(times, audio_feature.shape)
        sqrt_one_minus_alpha_cum = self.scheduler.get_sqrt_one_minus_alpha_cum(times, audio_feature.shape)

        # ----- x_t = \sqrt(alpha_t) * x_1 + \sqrt(1 - alpha_t) * epsilon ----- #
        true_noise = torch.randn_like(audio_feature)
        x_t = sqrt_alpha_cum * x_1 + sqrt_one_minus_alpha_cum * true_noise
        # print(f"predicting noise with input x_t of shape {x_t.shape}")
        # B, T, z
        # the model only condition on timestep t and perform denoising
        # we do not use it for conditional generation but use it as a speech normalizer
        if self.use_cond:
            pred_noise = self.model(
                x_t, times,
                prompt=src_feature,
                prompt_mask=src_mask,  # mask for self-attn
                input_mask=tgt_mask,  # mask for self-attn
                cond_drop_prob=0.1
            )
        else:
            pred_noise = self.model(
                x_t, times,
                input_mask=tgt_mask,  # mask for self-attn
                cond_drop_prob=0.1
            )

        snr = self.scheduler.get_snr(times)
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max=5.0)
        loss_weight = maybe_clipped_snr / snr
        noise_mask = tgt_mask.unsqueeze(2).expand(-1, -1, pred_noise.shape[2])
        noise_mse = F.mse_loss(pred_noise, true_noise, reduction='none')
        noise_mse[~noise_mask] = 0
        noise_mse = reduce(noise_mse, 'b ... -> b', 'mean')
        noise_mse = (noise_mse * loss_weight).mean()

        # ------- predict start and reconstruct back to representation and tokens ------#
        x_1_hat = safe_div(x_t - sqrt_one_minus_alpha_cum * pred_noise, sqrt_alpha_cum)
        x_1_decode, lm_pred = self.speech_decoder.decode_feature(x_1_hat, tgt_mask)
        select_mask = tgt_mask.unsqueeze(2).expand(-1, -1, x_1_decode.shape[2])
        recon_mse = self.mse_loss(x_1_decode[select_mask], audio[select_mask])
        # compute nll loss with target units
        lprobs = F.log_softmax(lm_pred, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        unit = audio_units.view(-1)
        unit_mask = unit.ne(0)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(unit_mask).eq(unit.masked_select(unit_mask))
        )
        total = torch.sum(unit_mask)
        acc = n_correct / total
        smooth_loss, _ = label_smoothed_nll_loss(
            lprobs,
            unit,
            0.1,
            ignore_index=0,
            reduce=True,
        )
        n_tokens = unit_mask.sum()
        smooth_loss = smooth_loss / n_tokens
        recon_loss = 50 * recon_mse + smooth_loss

        # scale recon by 1/T, as for noise mse, might need to boost it by some scale
        if self.multitask:
            all_loss = noise_mse + recon_loss / self.timesteps
        else:
            all_loss = noise_mse
        # print(f"recon mse: {recon_mse}, noise mse: {noise_mse}, nll loss: {smooth_loss}, acc: {acc}")

        loss_dict = {
            "total_loss": all_loss,
            "nll_loss": smooth_loss,
            "recon_mse_loss": recon_mse,
            "noise_loss": noise_mse,
            "acc": acc
        }
        return loss_dict
    