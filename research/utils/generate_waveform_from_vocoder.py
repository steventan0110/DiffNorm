# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
from pathlib import Path
import random
import soundfile as sf
import torch

from tqdm import tqdm
import fairseq
from fairseq import utils
from fairseq.models.text_to_speech.repr_hifigan import ReprHiFiGANModel
from fairseq.models.text_to_speech.unit_hifigan import UnitHiFiGANModel
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
import numpy as np


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, sample_id, pred_wav, suffix=""):
    sf.write(
        f"{args.results_path}/{sample_id}{suffix}_pred.wav",
        pred_wav.detach().cpu().numpy(),
        16000,
    )

def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out


def load_code(in_file, reduce, filter_score=None):
    out = []
    with open(in_file) as f:
        for line in f:
            sample_id, units = line.strip().split("|")
            if filter_score is not None:
                score = float(sample_id.split("=")[1])
                if score < filter_score:
                    continue
            units = units.split()
            units = process_units(units, reduce)
            units = list(map(int, units))
            out.append(units)
        # out = [list(map(int, process_units(line.strip().split(), reduce))) for line in f]
    return out

def repr_vocoder_gen(args):
    audio_to_evaluate = []
    with open(args.audio_id_file, "r") as f:
        for line in f:
            line = line.strip()
            # TODO: check language, for en, need to use mp3.wav as post-fix
            audio_path = f"{args.audio_dir}/{line}.mp3"
            audio_to_evaluate.append(audio_path)
    print(f"total audio to evaluate: {len(audio_to_evaluate)}")
    # load pretrained vocoder
    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    (
        model,
        cfg,
        task,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.vocoder]
    )
    model = model[0].to("cuda")
    model.eval()

    reader = HubertFeatureReader(
        checkpoint_path=args.mhubert_ckpt, layer=args.mhubert_layer
    )

    with torch.no_grad():
        for i, audio_path in enumerate(tqdm(audio_to_evaluate)):
            feat = reader.get_feats(audio_path) # T x 768
            feat = utils.move_to_cuda(feat)
            wave_pred = model.encoder.inference(feat.unsqueeze(0), src_lengths=None)
            wave_pred = wave_pred.squeeze(0)
            dump_result(args, i, wave_pred, suffix="")
            if i == args.limit:
                break
    print(f"done generating waveform for {len(audio_to_evaluate)} audio")
    exit(0)


def unit_vocoder_gen(args):
    data = load_code(args.in_code_file, args.reduce, filter_score=args.filter_score)

    (
        model,
        cfg,
        task,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.vocoder]
    )
    model = model[0].to("cuda")
    model.eval()
    model_dictionary = model.src_dict

    for i, d in tqdm(enumerate(data), total=len(data)):
        unit_str = " ".join([x for x in map(str, d)])
        unit_token = model_dictionary.encode_line(unit_str, append_eos=False, add_if_not_exist=False)
        x = unit_token.view(1, -1).to("cuda")
        wav = model.encoder.inference(x).squeeze(0)
        dump_result(args, i, wav, suffix="")
        if args.limit is not None and i >= args.limit:
            return
    exit(0)


def main(args):
    logger.info(args)
    use_cuda = torch.cuda.is_available() and not args.cpu
    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    if args.mode == "repr":
        # prepare audio wave and extract mhubert feature
        repr_vocoder_gen(args)
    else:
        # prepare unit sequence
        unit_vocoder_gen(args)



def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unit", "repr"], default="unit")
    parser.add_argument("--audio_dir", type=str, help="path to audio dir", default=None)
    parser.add_argument("--audio_id_file", type=str, help="path to audio id file", default=None)
    parser.add_argument("--mhubert_ckpt", type=str, help="path to the mhubert checkpoint", default=None)
    parser.add_argument("--mhubert_layer", type=int, default=11, help="mhubert layer to extract feature")
    parser.add_argument(
        "--in-code-file", type=str, help="one unit sequence per line"
    )
    parser.add_argument(
        "--vocoder", type=str, help="path to the CodeHiFiGAN vocoder"
    )
    parser.add_argument(
        "--vocoder-cfg",
        type=str,
        required=True,
        help="path to the CodeHiFiGAN vocoder config",
    )
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument(
        "--dur-prediction",
        action="store_true",
        help="enable duration prediction (for reduced/unique code sequences)",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=-1,
        help="Speaker id (for vocoder that supports multispeaker). Set to -1 to randomly sample speakers.",
    )
    parser.add_argument("--limit", type=int, default=None, help="limit the number of samples")
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    parser.add_argument("--reduce", action="store_true", help="reduce unit")
    parser.add_argument("--filter-score", type=int, default=None, help="generate high quality sample (cherrypick) over certain bleu score")
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
