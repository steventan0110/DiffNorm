# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F

from fairseq.data import ConcatDataset, Dictionary, FairseqDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import S2SDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
    _collate_frames,
    _is_int_or_np_int
)
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDataset
logger = logging.getLogger(__name__)



@dataclass
class ReprToSpeechDatasetItem(object):
    index: int
    audio: torch.Tensor
    feat: torch.Tensor


# unit-to speech dataset, where unit and speech are aligned (and are of the same lang)
class ReprToSpeechDataset(FairseqDataset):
    def __init__(self,
                 split: str,
                 is_train_split: bool,
                 cfg: S2SDataConfig,
                 audio_paths: List[str],
                 feat_paths: List[str],
                 n_frames: List[int],
                 src_langs: Optional[List[str]] = None,
                 ids: Optional[List[str]] = None,
                 ):

        self.split = split
        self.si_train_split = is_train_split
        self.cfg = cfg
        self.n_samples, self.n_frames = len(audio_paths), n_frames
        self.audio_paths = audio_paths
        self.feat_paths = feat_paths
        self.src_langs = src_langs
        self.ids = ids
        assert ids is None or len(ids) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        self.shuffle = cfg.shuffle if is_train_split else False
        # do not apply and feature transform/wave transform as we will use mhubert to
        # read in the audio later
        logger.info(self.__repr__())



    def __getitem__(self, index: int) -> ReprToSpeechDatasetItem:
        feat_path = self.feat_paths[index]
        feat_npy = np.load(feat_path)
        source = get_features_or_waveform(
            self.audio_paths[index],
            need_waveform=True,
            use_sample_rate=16000,
            waveform_transforms=None,
        )
        source = torch.from_numpy(source).float()
        feat = torch.from_numpy(feat_npy).float()

        # randomly subsample from the speech and feat into a 2-second clip
        assert self.n_frames[index] == feat.shape[0]
        max_pos = self.n_frames[index] - 100
        max_pos = min(max_pos, source.shape[0] // 320 - 100)
        pos = np.random.randint(0, max_pos+1)
        # print("random function called")
        selected_repr = feat[pos:pos+100]
        selected_audio = source[pos*320: (pos+100)*320] # 320 is the fixed upsampling rate used throughout the project
        # print(f"selected_repr shape: {selected_repr.shape}, selected_audio shape: {selected_audio.shape}")

        return ReprToSpeechDatasetItem(
            index=index,
            audio=selected_audio,
            feat=selected_repr
        )


    # overwrite based class function
    def __repr__(self):
        return (
                self.__class__.__name__
                + f'(split="{self.split}", n_samples={self.n_samples:_}, '
                  f"shuffle={self.shuffle}, "
                  f"feature_transforms=Not Enabled, "
                  f"waveform_transforms=Not Enabled, "
                  f"dataset_transforms=Not Enabled)"
        )

    def num_tokens(self, index):
        return 100 # hard code as we always use a chunk

    def size(self, index):
        return self.n_frames[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False

    def __len__(self):
        return self.n_samples


    def collater(
            self, samples: List[ReprToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        audios = torch.stack([x.audio for x in samples], dim=0) # B x T_raw
        feats = torch.stack([x.feat for x in samples], dim=0) # B x T x C
        batch_size, feat_len, feat_dim = feats.shape
        n_tokens = batch_size * feat_len
        feats_lengths = torch.tensor([feat_len] * batch_size, dtype=torch.long)
        audio_lengths = torch.tensor([x.shape[0] for x in audios], dtype=torch.long)

        net_input = {
            "src_tokens": feats,
            "src_lengths": feats_lengths,
            "prev_output_tokens": None,  # not used in NAT generation
            "tgt_speaker": None,  # not used
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": None,  # not used
            "target": audios,
            "target_lengths": audio_lengths,
            "ntokens": n_tokens,
            "nsentences": len(samples),
        }
        return out




class ReprToSpeechDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""
    KEY_SRC_FEAT = "src_feat"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
    ) -> SpeechToSpeechDataset:

        src_audio_paths = [
            s[cls.KEY_SRC_AUDIO].as_posix() for s in samples
        ]
        ids = [s[cls.KEY_ID] for s in samples]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        src_feat_paths = [s[cls.KEY_SRC_FEAT] for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        ds = ReprToSpeechDataset(
            split=split_name,
            is_train_split=is_train_split,
            cfg=data_cfg,
            audio_paths=src_audio_paths,
            feat_paths=src_feat_paths,
            n_frames=src_n_frames,
            src_langs=src_langs,
            ids=ids,
        )
        return ds

    @staticmethod
    def _load_samples_from_tsv(feat_root, raw_audio_root, split):
        manifest_file = f"{feat_root}/{split}.manifest.tsv"
        samples = []
        counter = 0
        with open(manifest_file) as f:
            f.readline() # skip the first line
            for line in f:
                feat_id, feat_len = line.rstrip().split("\t")
                # input is compressed 320x by mhubert, 2s audio -> 2 x 16000 / 320 = 100 frames. we need audio of at least 2s
                if int(feat_len) < 100:
                    continue
                audio_id = feat_id.split(".")[0]
                feat_file = Path(feat_root) / f"{split}/{feat_id}"
                if "/es-en/en" in feat_root or "/fr-en/en" in feat_root:
                    # dealing with en audio synthesis
                    audio_file = Path(raw_audio_root) /  f"{split}/{audio_id}.mp3.wav"
                else:
                    audio_file = Path(raw_audio_root) / f"{split}/{audio_id}.mp3"
                samples.append({
                    ReprToSpeechDatasetCreator.KEY_ID: audio_id,
                    ReprToSpeechDatasetCreator.KEY_SRC_AUDIO: audio_file,
                    ReprToSpeechDatasetCreator.KEY_SRC_FEAT: feat_file,
                    ReprToSpeechDatasetCreator.KEY_SRC_N_FRAMES: feat_len,
                })
                counter += 1
                if (not "train" in split) and counter > 4000:
                    # only keep 4k samples for dev/test purpose
                    break
        return samples

    @classmethod
    def from_tsv(
        cls,
        feat_root: str,
        raw_audio_root: str,
        data_cfg: S2SDataConfig,
        splits: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDataset:
        datasets = []
        for split in splits.split(","):
            samples = ReprToSpeechDatasetCreator._load_samples_from_tsv(feat_root, raw_audio_root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                data_cfg=data_cfg,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
