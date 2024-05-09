# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import torch
import time
from research.TranSpeech.dataset import SpeechToSpeechFastTranslateDatasetCreator
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
from argparse import Namespace
import json
logger = logging.getLogger(__name__)
from fairseq.utils import new_arange

@register_task("speech_to_speech_fasttranslate")
class SpeechToSpeechFastTranslateTask(SpeechToSpeechTask):
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = SpeechToSpeechFastTranslateDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.target_dictionary,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=self.multitask_tasks,
        )


    @staticmethod
    def create_gaussian_mask(target_lens, target_masks):
        bz = target_lens.size(0)
        max_len = target_lens.max().item() + 1 # +1 to handle eos
        # TODO: generalize these hyper-params?
        shift_low, shift_high = 0, target_lens // 6
        shift = torch.rand(bz, device=target_lens.device) * (shift_high - shift_low) + shift_low
        scale_low, scale_high = 2, 8 # this control how "spread out" the gaussian distribution is
        scale = torch.rand(bz, device=target_lens.device) * (scale_high - scale_low) + scale_low

        mean = target_lens / 2 - shift
        std_dev = target_lens / scale

        indices = torch.arange(max_len, device=target_lens.device).unsqueeze(0).expand(bz, -1)
        mean = mean.unsqueeze(1).expand(-1, max_len)
        std_dev = std_dev.unsqueeze(1).expand(-1, max_len)

        probs = torch.exp(-0.5 * ((indices - mean) / std_dev) ** 2)
        probs = probs / probs.max()
        random_scale = torch.rand(bz, 1, device=target_lens.device) + 0.5  # create randon scale between 0.5 and 1.5

        probs = probs * random_scale
        probs = torch.clamp(probs, 0, 1)
        masks = torch.bernoulli(probs).bool()
        masks = masks & target_masks
        return masks


    def inject_noise(self, target_tokens):
        def _side_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_lens = target_masks.sum(1)
            masks = self.create_gaussian_mask(target_lens, target_masks)
            prev_target_tokens = target_tokens.masked_fill(masks, unk)
            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        prob_for_side = random.Random().random()
        # max the uniform masking with bowl shape masking
        if self.args.use_side and prob_for_side > 0.5:
            return _side_mask(target_tokens)
        return _random_mask(target_tokens)

    def build_model(self, args, from_checkpoint=False):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_transformed_channels
        args.target_speaker_embed = self.data_cfg.target_speaker_embed is not None
        args.n_frames_per_step = self.args.n_frames_per_step

        model = super(SpeechToSpeechTask, self).build_model(args, from_checkpoint)

        if len(self.multitask_tasks) > 0:
            from research.TranSpeech.nar_transformer import (
                NARS2UTTransformerModel,
            )
            assert isinstance(model, NARS2UTTransformerModel)

        if self.args.eval_inference:
            self.eval_gen_args = json.loads(self.args.eval_args)
            self.generator = self.build_generator(
                [model], Namespace(**self.eval_gen_args)
            )

        return model

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def num_params(self, model, print_out=True, model_name="model"):
        import numpy as np
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        for task_name, task_obj in self.multitask_tasks.items():
            criterion.set_multitask_loss_weight(
                task_name, task_obj.args.get_loss_weight(update_num)
            )

        sample["net_input"]["prev_target"] = self.inject_noise(sample["net_input"]["target"])

        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        sample["net_input"]["prev_target"] = self.inject_noise(sample["net_input"]["target"])
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.args.eval_inference:
            hypos, inference_losses = self.valid_step_with_inference(
                sample, model, self.generator
            )
            for k, v in inference_losses.items():
                assert k not in logging_output
                logging_output[k] = v

        return loss, sample_size, logging_output

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from research.TranSpeech.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )
