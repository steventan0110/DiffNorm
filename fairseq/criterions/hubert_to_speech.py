# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from typing import List, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import lengths_to_mask
from fairseq.models.fairseq_model import FairseqEncoderModel
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


@register_criterion("unit_to_speech")
class UnitToSpeechLoss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model: FairseqEncoderModel, sample, reduction="mean", save_audio=False, is_training_disc=False, is_training=False):
        src_tokens = sample["net_input"]["src_tokens"]

        tgt_lens = sample["target_lengths"]
        tgt_tokens = sample["target"]
        tgt_duration = sample["target_duration"]
        duration_mask = tgt_duration.ne(-100) # -100 is hard-coded duration mask
        full_tgt_tokens = sample["full_target"]
        full_tgt_lengths = sample["full_target_lengths"]

        waveform, log_dur_out, loss_out = model(
            tgt_tokens, src_lengths=None,
            duration_label=tgt_duration,
            wave_true=src_tokens,
            full_target=full_tgt_tokens
        )

        loss_mel = loss_out["loss_mel"]
        loss_disc_f = loss_out["loss_disc_f"]
        loss_disc_s = loss_out["loss_disc_s"]
        loss_fm_f = loss_out["loss_fm_f"]
        loss_fm_s = loss_out["loss_fm_s"]
        loss_gen_f = loss_out["loss_gen_f"]
        loss_gen_s = loss_out["loss_gen_s"]

        # duration loss
        tgt_duration = tgt_duration.float()
        tgt_duration = tgt_duration.half() if log_dur_out.type().endswith(".HalfTensor") else tgt_duration
        # print(duration_mask)
        log_dur = torch.log(tgt_duration + 1)[duration_mask]
        dur_loss = F.mse_loss(log_dur_out[duration_mask], log_dur, reduction=reduction)


        # combine loss for generator and discriminator
        gen_loss = loss_gen_s + loss_gen_f + 2*loss_fm_s + 2*loss_fm_f
        disc_loss = loss_disc_s + loss_disc_f

        # auxillary metrics for best ckpt keeping
        si_snr_score = 0
        pesq_score = 0
        if is_training:
            if is_training_disc:
                # only used disc_loss will result in param unused error, we can only add in mel loss to bypass that stupid error..
                loss = disc_loss + loss_mel * 45 + dur_loss
            else:
                loss = gen_loss + loss_mel * 45 + dur_loss
        else:
            loss = gen_loss + loss_mel * 45 + disc_loss + dur_loss
            # only compute auxillary metrics when not training, to speech up training
            with torch.no_grad():
                pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
                si_snr = ScaleInvariantSignalNoiseRatio().to(waveform.device)
                try:
                    si_snr_score = si_snr(waveform, src_tokens).item()
                    pesq_score = pesq(waveform, src_tokens).item()
                except:
                    # pesq could give no utterance detected error sometimes
                    si_snr_score = 0
                    pesq_score = 0

        # prepare logging output
        sample_size = sample["nsentences"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "mel_loss": utils.item(loss_mel.data),
            "dur_loss": utils.item(dur_loss.data),
            "disc_loss": utils.item(disc_loss.data),
            "gen_loss": utils.item(gen_loss.data),
            "si_snr_score": si_snr_score,
            "pesq_score": pesq_score,
        }

        if save_audio:
            logging_output["waveform"] = waveform

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]
        for key in [
            "loss",
            "dur_loss",
            "mel_loss",
            "disc_loss",
            "gen_loss",
            "si_snr_score",
            "pesq_score"
        ]:
            vals = [log.get(key, 0) for log in logging_outputs]
            val = sum(val * w for val, w in zip(vals, ws))
            metrics.log_scalar(key, val, ntot, round=3)
        metrics.log_scalar("sample_size", ntot, len(logging_outputs))

        # inference metrics
        if "targ_frames" not in logging_outputs[0]:
            return
        n = sum(log.get("targ_frames", 0) for log in logging_outputs)
        for key, new_key in [
            ("mcd_loss", "mcd_loss"),
            ("pred_frames", "pred_ratio"),
            ("nins", "ins_rate"),
            ("ndel", "del_rate"),
        ]:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(new_key, val / n, n, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
