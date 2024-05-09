# inject DDPM sampler into the criterion

from typing import List, Dict, Any
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import lengths_to_mask
from fairseq.models.fairseq_model import FairseqEncoderModel
from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
)
from fairseq.models.text_to_speech.distributions import DiagonalGaussianDistribution
from fairseq.models.text_to_speech.diffusion import create_diffusion
@register_criterion("ddpm_discrete_loss")
class DDPMDiscreteLoss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.eps = 0.2 # TODO: make this configurable

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def forward(self, model: FairseqEncoderModel, sample, reduction="mean"):
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        tgt_feature = sample["reduce_target"]
        tgt_unit = sample["reduce_target_unit"]
        tgt_lengths = sample["reduce_target_lengths"]
        # we use non-reduced version as that's the one trained with vae model. And denoising is not sensitive to length reduction

        # tgt_feature = sample["target"]
        # tgt_unit = sample["target_unit"]
        # tgt_lengths = sample["target_lengths"]


        # print(tgt_unit.shape, tgt_feature.shape, src_tokens.shape)
        # ----------------------- DDPM loss computation ----------------------- #
        model_kwargs = dict(
            src_feature=src_tokens,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            unk_token=self.task.tgt_dict.unk_index,
        )
        loss_dict = model(tgt_feature, tgt_unit, **model_kwargs)
        loss = loss_dict["total_loss"]
        nll_loss = loss_dict["nll_loss"]
        mse_loss = loss_dict["recon_mse_loss"]
        noise_loss = loss_dict["noise_loss"]
        acc = loss_dict["acc"]
        sample_size = sample["nsentences"]
        logging_output = {
            "loss": utils.item(loss.data),
            "noise_loss": utils.item(noise_loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "mse_loss": utils.item(mse_loss.data),
            "acc": utils.item(acc.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]
        for key in [
            "loss",
            "noise_loss",
            "mse_loss",
            "nll_loss",
            "acc",
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
