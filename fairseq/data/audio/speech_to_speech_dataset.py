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

from fairseq.data import ConcatDataset, Dictionary
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

logger = logging.getLogger(__name__)


@dataclass
class SpeechToSpeechDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None

@dataclass
class UnitToSpeechDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    original_target: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    duration_label: Optional[torch.Tensor] = None


class SpeechToSpeechDataset(SpeechToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2SDataConfig,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        tgt_audio_paths: List[str],
        tgt_n_frames: List[int],
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
    ):
        tgt_texts = tgt_audio_paths if target_is_code else None
        super().__init__(
            split=split,
            is_train_split=is_train_split,
            cfg=data_cfg,
            audio_paths=src_audio_paths,
            n_frames=src_n_frames,
            ids=ids,
            tgt_dict=tgt_dict,
            tgt_texts=tgt_texts,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            n_frames_per_step=n_frames_per_step,
        )

        self.tgt_audio_paths = tgt_audio_paths
        self.tgt_lens = [t // self.n_frames_per_step for t in tgt_n_frames]

        assert not target_is_code or tgt_dict is not None
        self.target_is_code = target_is_code

        assert len(tgt_audio_paths) == self.n_samples
        assert len(tgt_n_frames) == self.n_samples

        self.tgt_speakers = None
        if self.cfg.target_speaker_embed:
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(
                self.cfg.target_speaker_embed, split
            )
            spk_emb_dict = {s["id"]: s["speaker_embed"] for s in samples}
            self.tgt_speakers = [spk_emb_dict[id] for id in self.ids]
            assert len(self.tgt_speakers) == self.n_samples

        logger.info(self.__repr__())

    def pack_units(self, input: torch.Tensor) -> torch.Tensor:
        if self.n_frames_per_step <= 1:
            # if =1 means the code is already reduced/packed
            return input

        offset = 4
        vocab_size = (
            len(self.tgt_dict) - offset
        )  # remove offset from <bos>, <pad>, <eos>, <unk>, which is specific to fairseq dictionary

        assert input.dim() == 1
        stacked_input = (
            input[:-1].view(-1, self.n_frames_per_step) - offset
        )  # remove <eos>
        scale = [
            pow(vocab_size, self.n_frames_per_step - 1 - i)
            for i in range(self.n_frames_per_step)
        ]
        scale = torch.LongTensor(scale).squeeze(0)
        res = input.new((len(input) - 1) // self.n_frames_per_step + 1).fill_(input[-1])
        res[:-1] = (stacked_input * scale).sum(dim=1) + offset

        return res

    def __getitem__(self, index: int) -> SpeechToSpeechDatasetItem:
        source = self._get_source_audio(index)

        tgt_lang_tag = None
        if self.cfg.prepend_tgt_lang_tag_as_bos:
            # prepend_tgt_lang_tag_as_bos: put tgt_lang_tag as bos of target
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)

        if not self.target_is_code:
            target = get_features_or_waveform(self.tgt_audio_paths[index])
            target = torch.from_numpy(target).float()
            target = self.pack_frames(target)
        else:
            target = self.tgt_dict.encode_line(
                self.tgt_audio_paths[index],
                add_if_not_exist=False,
                append_eos=True,
            ).long()
            if self.n_frames_per_step > 1:
                n_tgt_frame = target.size(0) - 1  # exclude <eos>
                keep_n_tgt_frame = n_tgt_frame - n_tgt_frame % self.n_frames_per_step
                target = torch.cat(
                    (
                        target[:keep_n_tgt_frame],
                        target.new_full((1,), self.tgt_dict.eos()),
                    ),
                    dim=0,
                )

        if self.tgt_speakers:
            tgt_spk = get_features_or_waveform(self.tgt_speakers[index])
            tgt_spk = torch.from_numpy(tgt_spk).float()
        else:
            tgt_spk = torch.FloatTensor([])
        return SpeechToSpeechDatasetItem(
            index=index,
            source=source,
            target=target,
            target_speaker=tgt_spk,
            tgt_lang_tag=tgt_lang_tag,
        )

    def _collate_target(self, samples: List[SpeechToSpeechDatasetItem]) -> torch.Tensor:
        if self.target_is_code:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            # convert stacked units to a single id
            pack_targets = [self.pack_units(x.target) for x in samples]
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                pack_targets,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            target_lengths = torch.tensor(
                [x.size(0) for x in pack_targets], dtype=torch.long
            )
        else:
            target = _collate_frames([x.target for x in samples], is_audio_input=False)
            bsz, _, d = target.size()
            prev_output_tokens = torch.cat(
                (target.new_full((bsz, 1, d), 0.0), target[:, :-1, :]), dim=1
            )
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            )

        return target, prev_output_tokens, target_lengths

    def collater(
        self, samples: List[SpeechToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, prev_output_tokens, target_lengths = self._collate_target(samples)
        target = target.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.target.size(0) for x in samples)

        tgt_speakers = None
        if self.cfg.target_speaker_embed:
            tgt_speakers = _collate_frames(
                [x.target_speaker for x in samples], is_audio_input=True
            ).index_select(0, order)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
            "tgt_speaker": tgt_speakers,  # TODO: unify "speaker" and "tgt_speaker"
        }

        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": tgt_speakers,  # to support Tacotron2 loss for speech-to-spectrogram model
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out


class SpeechToSpeechMultitaskDataset(SpeechToSpeechDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]:
        s2s_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2s_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out


class SpeechToSpeechDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDataset:
        audio_root = Path(data_cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = []

        src_audio_paths = [
            (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples
        ]



        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = (
            SpeechToSpeechMultitaskDataset if has_multitask else SpeechToSpeechDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            target_is_code=target_is_code,
            tgt_dict=tgt_dict,
            n_frames_per_step=n_frames_per_step,
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
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
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                data_cfg=data_cfg,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                multitask=multitask,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]



# unit-to speech dataset, where unit and speech are aligned (and are of the same lang)
class UnitToSpeechDataset(SpeechToSpeechDataset):
    def _get_source_audio(self, index: Union[int, List[int]]) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms
        applied. For ConcatAug, source audios for given indices are
        concatenated in given order.
        Args:
            index (int or List[int]): index—or in the case of ConcatAug,
            indices—to pull the source audio for
        Returns:
            source audios concatenated for given indices with
            relevant transforms appplied
        """
        if _is_int_or_np_int(index):
            source = get_features_or_waveform(
                self.audio_paths[index],
                need_waveform=True,
                use_sample_rate=self.cfg.use_sample_rate,
                waveform_transforms=None,
            )
        else:
            source = np.concatenate(
                [
                    get_features_or_waveform(
                        self.audio_paths[i],
                        need_waveform=True,
                        use_sample_rate=self.cfg.use_sample_rate,
                        waveform_transforms=None,
                    )
                    for i in index
                ]
            )

        source = torch.from_numpy(source).float()
        # if self.cfg.standardize_audio:
        #     with torch.no_grad():
        #         source = F.layer_norm(source, source.shape)
        return source

    def _reduce_tgt(self, tokens):
        dedup_tokens = []
        duration_label = []
        accu_duration = 1
        for i, token in enumerate(tokens):
            if i == 0:
                dedup_tokens.append(token)
            else:
                if token != tokens[i - 1]:
                    # triggering a new duration counter
                    duration_label.append(accu_duration)
                    dedup_tokens.append(token)
                    accu_duration = 1
                else:
                    # duplicates found
                    accu_duration += 1
        # append the last duration
        duration_label.append(accu_duration)
        return dedup_tokens, duration_label

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

    def __getitem__(self, index: int) -> UnitToSpeechDatasetItem:
        source = self._get_source_audio(index)
        tgt_lang_tag = None
        # tgt audio is non-reduced
        tgt_tokens_str = self.tgt_audio_paths[index]
        tgt_tokens = tgt_tokens_str.split(" ")
        # size of source speech is about 320x of len(tgt_tokens), though offset exists
        token_len = len(tgt_tokens)
        max_pos = token_len - 100

        # sub-select audio clips of 2s
        max_pos = min(max_pos, source.shape[0] // 320 - 100)
        pos = np.random.randint(0, max_pos + 1)
        selected_tokens = tgt_tokens[pos:pos + 100]
        selected_audio = source[pos * 320: (pos + 100) * 320]

        # construct actual tokens
        reduce_tgt, duration_label = self._reduce_tgt(selected_tokens)
        reduce_tgt_str = " ".join(reduce_tgt)
        selected_tgt_str = " ".join(selected_tokens)
        target = self.tgt_dict.encode_line(
            reduce_tgt_str,
            add_if_not_exist=False,
            append_eos=False,
        ).long()
        original_target = self.tgt_dict.encode_line(
            selected_tgt_str,
            add_if_not_exist=False,
            append_eos=False,
        ).long()  # this is un-reduced version


        duration_label = torch.tensor(duration_label, dtype=torch.long)
        tgt_spk = torch.FloatTensor([])
        return UnitToSpeechDatasetItem(
            index=index,
            source=selected_audio,
            target=target,
            original_target=original_target,
            target_speaker=tgt_spk,
            tgt_lang_tag=tgt_lang_tag,
            duration_label=duration_label,
        )

    def collater(
            self, samples: List[UnitToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        # frames, n_frames = self._collate_audio_frames([x.source for x in samples], True)
        """
        we have made each audio sequence and original tgt have the same shape, no need to pad anymore
        only need to take care of padding for duration labels

        UnitToSpeechDatasetItem(
            index=index,
            source=source,
            target=target,
            original_target=original_target,
            target_speaker=tgt_spk,
            tgt_lang_tag=tgt_lang_tag,
            duration_label=duration_label,
        )
        """
        frames = torch.stack([x.source for x in samples], dim=0) # B x T_raw=32000
        original_target = torch.stack([x.original_target for x in samples], dim=0) # B x T
        target = fairseq_data_utils.collate_tokens(
            [x.target for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        duration_labels = fairseq_data_utils.collate_tokens(
            [x.duration_label for x in samples],
            -100,  # use -100 as duration padding
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        batch_size, target_len = original_target.shape
        ntokens = batch_size * target_len
        target_lengths = target.ne(self.tgt_dict.pad()).sum(dim=1)
        original_target_lengths = original_target.ne(self.tgt_dict.pad()).sum(dim=1)
        net_input = {
            "src_tokens": frames,
            "src_lengths": None,
            "prev_output_tokens": None,  # not used in NAT generation
            "tgt_speaker": None,  # not used
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": None,  # not used
            "target": target,
            "full_target": original_target,
            "target_lengths": target_lengths,
            "full_target_lengths": original_target_lengths,
            "target_duration": duration_labels,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out




class UnitToSpeechDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDataset:
        audio_root = Path(data_cfg.audio_root) # it is .
        # TODO: replace this hacky code with more general solution
        # the tsv is prepared for speech-to-speech, which means the src_audio is in another language. We forcely correct the language id here.
        # example: 'src_audio': '/scr-ssd/chenyuz/CVSS/es-en/en/dev/common_voice_es_19610400.mp3.wav' -> 'src_audio': '/scr-ssd/chenyuz/CVSS/es-en/es/dev/common_voice_es_19610400.mp3'

        # src_audio_paths = [s[cls.KEY_SRC_AUDIO] for s in samples]
        src_audio_paths = [
            (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples
        ]
        ids = [s[cls.KEY_ID] for s in samples]
        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = UnitToSpeechDataset
        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            target_is_code=target_is_code,
            tgt_dict=tgt_dict,
            n_frames_per_step=n_frames_per_step,
        )
        return ds

    @staticmethod
    def _load_samples_from_tsv(root, split):
        # quant tsv is file of format: audio_id|units
        tsv_path = Path(root) / f"{split}.quant.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        samples = []
        counter = 0
        with open(tsv_path) as f:
            for line in f:
                audio_id, units = line.rstrip().split("|")
                tgt_len = len(units.split())
                if tgt_len < 100:
                    # at leaset 100 tokens => 2s audio clip
                    continue
                if ".mp3" in audio_id:
                    audio_id = audio_id.split(".")[0]
                if "/es-en/en" in root or "/fr-en/en" in root:
                    # dealing with en audio synthesis
                    audio_file = Path(root) /  f"{split}/{audio_id}.mp3.wav"
                else:
                    audio_file = Path(root) / f"{split}/{audio_id}.mp3"
                samples.append({
                    UnitToSpeechDatasetCreator.KEY_ID: audio_id,
                    UnitToSpeechDatasetCreator.KEY_SRC_AUDIO: audio_file,
                    UnitToSpeechDatasetCreator.KEY_SRC_N_FRAMES: tgt_len,
                    UnitToSpeechDatasetCreator.KEY_TGT_AUDIO: units,
                    UnitToSpeechDatasetCreator.KEY_TGT_N_FRAMES: tgt_len,
                })
                counter += 1
                # for debugging purpose
                # if counter == 1000:
                #     break
                if (not "train" in split) and counter > 4000:
                    # only keep 4k samples for dev/test purpose
                    break
        return samples

    @classmethod
    def from_tsv(
        cls,
        root: str,
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
            samples = UnitToSpeechDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                data_cfg=data_cfg,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                multitask=multitask,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
