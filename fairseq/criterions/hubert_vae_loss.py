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


@register_criterion("hubert_vae_loss")
class HubertVAELoss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.beta = self.task.beta
        print(f"beta: {self.beta}")

    def forward(self, model: FairseqEncoderModel, sample, reduction="mean", save_audio=False, is_training_disc=False, is_training=False):
        # B, T=100, C=768
        src_tokens = sample["net_input"]["src_tokens"] # all have the same length = 100 (2s audio)
        loss_out = model(
            src_tokens,
            src_lengths=None,
        )
        recon_loss = loss_out["recon_loss"]
        kl_loss = loss_out["kl_loss"]
        mean = loss_out["mean"]
        std = loss_out["std"]
        # disc_loss = loss_out["disc_loss"]
        loss = 10 * recon_loss + self.beta * kl_loss
        # if is_training:
        #     if is_training_disc:
        #         loss = disc_loss + recon_loss + 0.1 * kl_loss
        #     else:
        #         loss = gen_loss + 0.1 * kl_loss + 10 * recon_loss
        # else:
        #     loss = recon_loss + 0.1 * kl_loss + disc_loss + gen_loss

        # prepare logging output
        sample_size = sample["nsentences"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean": mean,
            "std": std,
            # "disc_loss": disc_loss.item(),
            # "gen_loss": gen_loss.item(),
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]
        for key in [
            "loss",
            "recon_loss",
            "kl_loss",
            "mean",
            "std"
            # "disc_loss",
            # "gen_loss",
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
