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
@register_criterion("ddpm_latent_loss")
class DDPMLatentLoss(FairseqCriterion):
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
        tgt_lengths = sample["target_lengths"]
        tgt_sample = sample["target"]

        # ----------------------- DDPM loss computation ----------------------- #
        model_kwargs = dict(
            src_feature=src_tokens,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths
        )
        loss, misc = model(tgt_sample, **model_kwargs)

        # compute CE loss for length prediction
        # length_out = loss_dict["misc"]["quantity_loss"]
        # # fake_loss = loss_dict["misc"]["fake_loss"]
        # target_len = tgt_lengths.clamp(min=0, max=127)
        # length_loss, _ = self.compute_loss(
        #     model, [length_out], {'target': target_len}, reduce=True)
        # length_loss = length_loss / src_tokens.shape[0]
        # loss_noise = loss_dict["loss"].mean()
        # mse_loss = loss_dict["mse"].mean()
        # vb_loss = loss_dict["vb"].mean()
        # loss_noise = self.loss_fn(epsilon_theta, noise)
        # print("noise loss:", loss_noise.item())
        # latent_pad = tgt_mask.unsqueeze(1).unsqueeze(3).expand_as(epsilon_theta)
        # loss_noise = self.loss_fn(epsilon_theta[latent_pad], noise[latent_pad])
        # loss = loss_noise + 0.1 * length_loss
        # prepare logging output
        # print(loss, loss_noise, length_loss)

        sample_size = sample["nsentences"]
        logging_output = {
            "loss": utils.item(loss.data),
            # "noise_loss": utils.item(loss_noise.data),
            # "length_loss": utils.item(length_loss.data),
            # "mse_loss": utils.item(mse_loss.data),
            # "vb_loss": utils.item(vb_loss.data),
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
            # "noise_loss",
            # "length_loss",
            # "mse_loss",
            # "vb_loss"
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
