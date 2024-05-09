# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import MultitaskConfig, S2SDataConfig
from fairseq.data.audio.speech_to_speech_dataset import UnitToSpeechDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion
from fairseq.optim.amp_optimizer import AMPOptimizer
logger = logging.getLogger(__name__)


@register_task("unit_to_speech")
class UnitToSpeechTask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--target-is-code",
            action="store_true",
            help="set if target is discrete unit instead of spectrogram",
        )
        parser.add_argument(
            "--target-code-size", type=int, default=None, help="# discrete units"
        )
        parser.add_argument("--save-audio", action="store_true")
        parser.add_argument(
            "--n-frames-per-step",
            type=int,
            default=1,
            help="# stacked frames, use 1 for reduced discrete unit sequence",
        )
        parser.add_argument("--eval-inference", action="store_true")
        parser.add_argument(
            "--eval-args",
            type=str,
            default="{}",
            help='generation args for speech-to-unit model , e.g., \'{"beam": 5, "max_len_a": 1}\', as JSON string',
        )
        parser.add_argument("--eos-prob-threshold", type=float, default=0.5)
        parser.add_argument(
            "--mcd-normalize-type",
            type=str,
            default="targ",
            choices=["targ", "pred", "path"],
        )
        parser.add_argument(
            "--vocoder",
            type=str,
            default="griffin_lim",
            choices=["griffin_lim", "hifigan", "code_hifigan"],
        )
        parser.add_argument("--spec-bwd-max-iter", type=int, default=8)
        parser.add_argument(
            "--infer-target-lang",
            type=str,
            default="",
            help="target language for inference",
        )
        parser.add_argument(
            "--dummy-config",
            type=str
        )
        parser.add_argument(
            "--vocoder-config",
            type=str)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.src_dict = tgt_dict # as the only dict
        if args.dummy_config.startswith("/scr-ssd/"):
            args.dummy_config = args.dummy_config.replace("/scr-ssd/chenyuz", "/weka/scratch/jzhan237/diff_s2s")
        self.data_cfg = S2SDataConfig(Path(args.dummy_config))
        # self.data_cfg = None # not used
        self.multitask_tasks = {}
        self.tgt_dict_mt = None
        self.eos_token_mt = None
        self._infer_tgt_lang_id = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        # data_cfg = S2SDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict = None
        infer_tgt_lang_id = None
        if args.target_is_code:
            assert args.target_code_size is not None
            tgt_dict = Dictionary() # unit based dictionary
            for i in range(args.target_code_size):
                tgt_dict.add_symbol(str(i))
        logger.info(f"dictionary size: " f"{len(tgt_dict):,}")

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        assert args.n_frames_per_step >= 1
        assert (
            not args.eval_inference
            or (args.target_is_code and args.vocoder == "code_hifigan")
            or (not args.target_is_code and args.vocoder != "code_hifigan")
        )

        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = UnitToSpeechDatasetCreator.from_tsv(
            root=self.args.data,
            data_cfg=self.data_cfg,
            splits=split,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.target_dictionary,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=self.multitask_tasks,
        )

    def has_sharded_data(self, split):
        """
        we force dataset to be re-loaded every epoch
        pretending to be loaded from sharded data will disable iterator cache
        and allow random init of dataset (which is useful to get random chunk of speech during training
        """
        return "train" in split

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def target_dictionary_mt(self):
        return self.tgt_dict_mt

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args, from_checkpoint=False):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_transformed_channels
        args.target_speaker_embed = self.data_cfg.target_speaker_embed is not None
        args.n_frames_per_step = self.args.n_frames_per_step
        model = super().build_model(args, from_checkpoint)
        return model


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        # print("current train step update number: ", update_num)
        update_group = model.get_groups_for_update(update_num)
        train_disc = update_group == "discriminator"
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, is_training_disc=train_disc, is_training=True)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()

        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, save_audio=self.args.save_audio)
        # we also need to prepare audios

        # TODO: eval_inference is not supported for now
        if self.args.eval_inference:
            hypos, inference_losses = self.valid_step_with_inference(
                sample, model, self.generator
            )
            for k, v in inference_losses.items():
                assert k not in logging_output
                logging_output[k] = v

        return loss, sample_size, logging_output

    def valid_step_with_inference(self, sample, model, generator):
        if self.args.target_is_code:
            hypos = generator.generate([model], sample)
            tgt_lens = (
                sample["target_lengths"] - 1
            ) * self.args.n_frames_per_step  # strip <eos>
            for b, (f, l) in enumerate(zip(sample["target"], tgt_lens)):
                hypos[b][0]["targ_waveform"] = self.vocoder(
                    {"code": f[:l] - 4},  # remove <bos>, <pad>, <eos>, <unk>
                    dur_prediction=self.eval_gen_args.get("dur_prediction", False),
                )
                if len(hypos[b][0]["tokens"]) > 0:
                    hypos[b][0]["waveform"] = self.vocoder(
                        {"code": hypos[b][0]["tokens"] - 4},
                        dur_prediction=self.eval_gen_args.get("dur_prediction", False),
                    )
                else:
                    hypos[b][0]["waveform"] = torch.flip(
                        hypos[b][0]["targ_waveform"], dims=[0]
                    )
        else:
            hypos = [
                [hypo] for hypo in generator.generate(model, sample, has_targ=True)
            ]

        losses = {
            "mcd_loss": 0.0,
            "targ_frames": 0.0,
            "pred_frames": 0.0,
            "path_frames": 0.0,
            "nins": 0.0,
            "ndel": 0.0,
        }
        rets = batch_mel_cepstral_distortion(
            [hypo[0]["targ_waveform"] for hypo in hypos],
            [hypo[0]["waveform"] for hypo in hypos],
            self.data_cfg.output_sample_rate,
            normalize_type=None,
        )
        for d, extra in rets:
            pathmap = extra[-1]
            losses["mcd_loss"] += d.item()
            losses["targ_frames"] += pathmap.size(0)
            losses["pred_frames"] += pathmap.size(1)
            losses["path_frames"] += pathmap.sum().item()
            losses["nins"] += (pathmap.sum(dim=1) - 1).sum().item()
            losses["ndel"] += (pathmap.sum(dim=0) - 1).sum().item()
        losses["norm_frames"] = losses[
            f"{getattr(self.args, 'mcd_normalize_type', 'targ')}_frames"
        ]

        return hypos, losses

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return super().inference_step(
                generator,
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )


    def optimizer_step(self, optimizer, model, update_num):
        if hasattr(model, "get_groups_for_update"):
            groups = model.get_groups_for_update(update_num)
            optimizer.step(groups={groups})
        else:
            optimizer.step()