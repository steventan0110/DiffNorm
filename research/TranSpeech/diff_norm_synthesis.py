# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import soundfile as sf
import torch
from dataclasses import dataclass
from tqdm import tqdm
import fairseq
from fairseq import utils
import numpy as np
from fairseq.data.data_utils import lengths_to_mask

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def reduce_token(tokens):
    dedup_tokens = []
    duration_label = []
    index_to_keep = []
    accu_duration = 1
    for i, token in enumerate(tokens):
        if i == 0:
            dedup_tokens.append(token)
            index_to_keep.append(i)
        else:
            if token != tokens[i - 1]:
                # triggering a new duration counter
                duration_label.append(accu_duration)
                dedup_tokens.append(token)
                index_to_keep.append(i)
                accu_duration = 1
            else:
                # duplicates found
                accu_duration += 1
    # append the last duration
    duration_label.append(accu_duration)
    return dedup_tokens, duration_label, torch.Tensor(index_to_keep).long()


@dataclass
class DataItem:
    audio_id: str
    src_audio: str
    src_n_frames: int
    reduce_tgt_unit: str
    reduce_tgt_n_frames: int


@dataclass
class AllDataItem:
    audio_id: str
    src_audio: str
    src_n_frames: int
    tgt_unit: str
    tgt_n_frames: int
    reduce_tgt_unit: str
    reduce_tgt_n_frames: int
    feature_file: str


def prepare_data(args, split):
    data = []
    batch_size = 100

    # ------------ load the mapping for reduce and original unit directory ------------#
    reduce_tsv_dir = f"{args.reduce_tsv_dir}/{split}.tsv"
    orig_tsv_dir = f"{args.orig_tsv_dir}/{split}.tsv"
    feature_dir = f"{args.feature_dir}/{split}"
    reduce_map = {}
    with open(reduce_tsv_dir, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            if len(line) != 5:
                continue
            audio_id, src_audio, src_n_frames, tgt_audio, tgt_n_frames = line
            src_n_frames = int(src_n_frames)
            tgt_n_frames = int(tgt_n_frames)
            reduce_map[audio_id] = DataItem(audio_id, src_audio, src_n_frames, tgt_audio, tgt_n_frames)

    unfound = 0
    full_map = {}
    with open(orig_tsv_dir, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            if len(line) != 5:
                continue
            audio_id, src_audio, src_n_frames, tgt_audio, tgt_n_frames = line
            tgt_n_frames = int(tgt_n_frames)
            if audio_id not in reduce_map:
                unfound += 1
                continue
            feature_file = f"{feature_dir}/{audio_id}.feat.npy"
            if not Path(feature_file).exists():
                unfound += 1
                continue
            data_item = reduce_map[audio_id]
            full_data_item = AllDataItem(
                audio_id,
                data_item.src_audio, data_item.src_n_frames,
                tgt_audio, tgt_n_frames,
                data_item.reduce_tgt_unit, data_item.reduce_tgt_n_frames,
                feature_file
            )
            full_map[audio_id] = full_data_item

    print("Unfound: ", unfound)
    # ------------- start preparing batches of data for diffusion model --------------#

    cur_batch, all_batch = [], []
    for data_item in full_map.values():
        if len(cur_batch) == batch_size:
            all_batch.append(cur_batch)
            cur_batch = []
        cur_batch.append(data_item)
    if len(cur_batch) > 0:
        all_batch.append(cur_batch)
    print("Prepared {} batches".format(len(all_batch)))
    return all_batch


def prepare_batch_data(batch_data):
    # process the list of data and pad them
    all_ids = []
    all_tgt_feat = []
    tgt_lengths = []
    all_units = []
    all_src_audio, all_src_n_frames = [], []
    for data_item in batch_data:
        all_ids.append(data_item.audio_id)
        all_src_audio.append(data_item.src_audio)
        all_src_n_frames.append(data_item.src_n_frames)
        tgt_feat_path = data_item.feature_file
        full_unit, reduce_unit = data_item.tgt_unit, data_item.reduce_tgt_unit
        tgt_feat = torch.from_numpy(np.load(tgt_feat_path)).float().to("cuda")
        full_unit = [int(_) for _ in full_unit.split(" ")]
        reduce_unit = [int(_) for _ in reduce_unit.split(" ")]

        # learn unit mapping from processed full and reduce unit to get our selected repr
        _, _, index_to_keep = reduce_token(full_unit)
        selected_tgt_feat = tgt_feat[index_to_keep]
        assert selected_tgt_feat.shape[0] == len(reduce_unit) == data_item.reduce_tgt_n_frames
        tgt_len = selected_tgt_feat.shape[0]

        tgt_lengths.append(tgt_len)
        all_tgt_feat.append(selected_tgt_feat)
        all_units.append(torch.Tensor(reduce_unit).long().to("cuda"))

    # pad the data
    tgt_lengths = torch.Tensor(tgt_lengths).long().to("cuda")
    max_tgt_len = tgt_lengths.max().item()
    batch_size = len(batch_data)

    padded_tgt_feature = torch.zeros(batch_size, max_tgt_len, 768).to("cuda")
    padded_units = torch.zeros(batch_size, max_tgt_len).to("cuda").long()
    for i in range(batch_size):

        padded_tgt_feature[i, :tgt_lengths[i], :] = all_tgt_feat[i]
        padded_units[i, :tgt_lengths[i]] = all_units[i]

    return all_ids, all_src_audio, all_src_n_frames, padded_tgt_feature, padded_units, tgt_lengths




def main(args):
    logger.info(args)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # prepare diffusion model
    (
        model,
        cfg,
        task,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.model_ckpt]
    )

    model = model[0].to("cuda")
    model.eval()


    for split in ["test", "dev", "train"]:
    # for split in ["test"]:
        data = prepare_data(args, split)
        out_file = f"{args.output_dir}/{split}.tsv"
        write_handle = open(out_file, "w")
        write_handle.write("id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n")

        for i, batch_data in tqdm(enumerate(data), total=len(data)):
            audio_ids, src_audios, src_n_frames, tgt_feat, ref_units, tgt_lengths = prepare_batch_data(batch_data)
            tgt_mask = lengths_to_mask(tgt_lengths)
            # prepare the initial input
            pred_units, _, _ = model.encoder.ddim_sample(
                tgt_feat,
                input_mask=tgt_mask,
                cond_scale=1.0, # no cond is used
                ref_units=ref_units,
                start_step=args.start_step
            )
            for j, diff_unit in enumerate(pred_units):
                tgt_n_frame = diff_unit.shape[0]
                diff_unit = diff_unit.cpu().numpy()
                diff_unit_str = diff_unit.tolist()
                # print(diff_unit_str)
                diff_unit_str, _, _ = reduce_token(diff_unit_str)
                audio_id = audio_ids[j]
                src_audio = src_audios[j]
                src_n_frame = src_n_frames[j]
                tgt_audio = " ".join([str(_) for _ in diff_unit_str])
                line_to_write = f"{audio_id}\t{src_audio}\t{src_n_frame}\t{tgt_audio}\t{tgt_n_frame}"
                print(line_to_write, file=write_handle)

        write_handle.close()
        print("Finished processing ", split)
        # break

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduce_tsv_dir", type=str, help="path to reduced tsv dir", default=None)
    parser.add_argument("--orig_tsv_dir", type=str, help="path to original tsv dir", default=None)
    parser.add_argument("--dummy-config", type=str, help="path to a dummy config file", default=None)
    parser.add_argument("--feature_dir", type=str, help="path to target vae feats", default=None)
    parser.add_argument("--model_ckpt", type=str, help="path to the diffusion model checkpoint", default=None)
    parser.add_argument("--start_step", type=int, default=50)
    parser.add_argument("--output_dir", type=str, help="path to output dir")
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
