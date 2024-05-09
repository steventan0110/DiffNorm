# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import soundfile as sf
import torch

from tqdm import tqdm
import fairseq
from fairseq import utils
import numpy as np
import joblib
import copy
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_mask
from fairseq.utils import new_arange
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
import sacrebleu
from fairseq.data.audio.audio_utils import get_features_or_waveform

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, sample_id, pred_wav, suffix=""):
    file_name = f"{args.output_dir}/{sample_id}{suffix}_pred.wav"
    sf.write(
        f"{args.output_dir}/{sample_id}{suffix}_pred.wav",
        pred_wav.detach().cpu().numpy(),
        16000,
    )


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")



def vocoder_inference(args, id, suffix, vocoder_model, audio_feature):
    # audio_feature: B, T, C
    audio_feature = audio_feature - 4
    x = {
        "code": audio_feature.view(1, -1),
    }
    wave_pred = vocoder_model(x, dur_prediction=True)
    wave_pred = wave_pred.squeeze(0)
    dump_result(args, id, wave_pred, suffix=suffix)

def unit_to_str(units):
    np_unit = units.squeeze().cpu().numpy()
    out = [str(np_unit[i]) for i in range(np_unit.shape[0])]
    out_str = " ".join(out)
    return out_str




def prepare_batch(batch, feature_transforms, use_ref_len=False):
    audio_files, _, _, hyp_units, ref_units = zip(*batch)
    src_lengths, hyp_lengths = [], []
    ref_lengths = []
    all_src_audio, all_hyp_units, all_ref_units = [], [], []
    for audio_file, _, _, hyp_unit, ref_unit in batch:
        source = get_features_or_waveform(
            audio_file,
            need_waveform=False,
            use_sample_rate=16000,
            waveform_transforms=None,
        )
        source_audio = feature_transforms(source)
        source_audio = torch.from_numpy(source_audio).float().to("cuda").unsqueeze(0)
        # map unit str to numbers
        hyp_unit = [int(x) for x in hyp_unit.split(" ")]
        ref_unit = [int(x) for x in ref_unit.split(" ")]
        hyp_unit = np.array(hyp_unit).astype(int)
        ref_unit = np.array(ref_unit).astype(int)
        hyp_unit = torch.from_numpy(hyp_unit).long().to("cuda").unsqueeze(0)
        ref_unit = torch.from_numpy(ref_unit).long().to("cuda").unsqueeze(0)
        src_lengths.append(source_audio.shape[1])
        hyp_lengths.append(hyp_unit.shape[1])
        ref_lengths.append(ref_unit.shape[1])
        all_src_audio.append(source_audio)
        all_hyp_units.append(hyp_unit)
        all_ref_units.append(ref_unit)
    src_lengths = torch.tensor(src_lengths).to("cuda")
    hyp_lengths = torch.tensor(hyp_lengths).to("cuda")
    ref_lengths = torch.tensor(ref_lengths).to("cuda")
    padded_src_audio = src_lengths.new_full((len(all_src_audio), src_lengths.max(), 80), 0).float()
    length_to_use = ref_lengths if use_ref_len else hyp_lengths
    padded_hyp_units = hyp_lengths.new_full((len(all_hyp_units), length_to_use.max()+1), 1) # pad token is 1
    units_to_use = all_ref_units if use_ref_len else all_hyp_units
    for i, cur_unit in enumerate(units_to_use):
        padded_hyp_units[i, :cur_unit.shape[1]] = 3 # fill in mask token
        padded_hyp_units[i, cur_unit.shape[1]] = 2 # eos token is 2

    for i, src_audio in enumerate(all_src_audio):
        padded_src_audio[i, :src_audio.shape[1]] = src_audio
    return padded_hyp_units, padded_src_audio, src_lengths, length_to_use + 1





@torch.no_grad()
def gen_with_units(args):
    # prepare data path for evaluation
    batch_size = 100
    all_batches = []
    with open(args.hyp_unit_file, "r") as f:
        hyp_units = f.read().split("\n")
    with open(args.ref_unit_file, "r") as f:
        ref_units = f.read().split("\n")
    assert len(hyp_units) == len(ref_units)
    with open(args.audio_id, "r") as f:
        cur_batch = []
        for i, line in enumerate(f):
            line = line.strip()
            src_feat_path = f"{args.src_feat_dir}/{line}.feat.npy"  # assume source is en,
            tgt_feat_path = f"{args.tgt_feat_dir}/{line}.feat.npy"
            audio_file = f"{args.audio_dir}/{line}.mp3.wav"
            hyp_unit = hyp_units[i]
            ref_unit = ref_units[i]
            cur_batch.append((audio_file, src_feat_path, tgt_feat_path, hyp_unit, ref_unit))
            if len(cur_batch) == batch_size:
                all_batches.append(cur_batch)
                cur_batch = []
        if len(cur_batch) > 0:
            all_batches.append(cur_batch)

    logger.info(f"Loading K-means model from {args.mhubert_ckpt} ...")
    kmeans_model = joblib.load(open(args.mhubert_ckpt, "rb"))
    kmeans_model.verbose = False
    (
        model,
        cfg,
        task,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.model_ckpt]
    )

    model = model[0].to("cuda")
    model.cg_prob = 0
    model.eval()
    # print(model.cg_prob)

    feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
        task.data_cfg.get_feature_transforms("test", False)
    )
    from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
    import json
    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder_model = CodeHiFiGANVocoder(args.vocoder_ckpt, vocoder_cfg)
    vocoder_model = vocoder_model.to("cuda")
    file_number = 0
    unk_token = model.unk
    pad_token = model.pad
    steps = args.num_iter
    cg_scale = args.cg_scale
    use_cg = False
    if cg_scale > 0:
        use_cg = True
    use_sp = args.use_sp
    if use_sp:
        assert not use_cg
    print("#" * 100)
    print(f"Start generating with {steps} steps, use_cg={use_cg}, cg_scale={cg_scale}, use_sp={use_sp} ...")
    print("#" * 100)

    for batch_id, cur_batch in tqdm(enumerate(all_batches), total=len(all_batches)):
        if args.limit is not None and file_number >= args.limit:
            break
        output_tokens, src_audio, src_lengths, tgt_lengths = prepare_batch(cur_batch, feature_transforms, use_ref_len=False)
        encoder_out = model.forward_encoder([src_audio, src_lengths])
        encoder_out_for_cg = copy.deepcopy(encoder_out) if use_cg else None
        # ----------------- Use NAT model for Inference ----------------------- #
        for step_id in range(steps):
            output_scores = output_tokens.new_full(output_tokens.shape, 0.0).float()
            output_masks = output_tokens == unk_token

            if use_sp and step_id < 0:
                # K=2 for y0 initialization following the original paper
                encoder_out_for_sp = copy.deepcopy(encoder_out)
                orig_logits, extra = model.decoder.sp_forward(
                    normalize=False,
                    prev_output_tokens=output_tokens,  # [B, T]
                    encoder_out=encoder_out_for_sp,
                )
            else:
                orig_logits, extra = model.decoder(
                    normalize=False,
                    prev_output_tokens=output_tokens,  # [B, T]
                    encoder_out=encoder_out,
                    inference_mode=True,
                )

            if use_cg:
                # classifier free guidance
                cg_logits, _ = model.decoder.cg_forward(
                    normalize=False,
                    prev_output_tokens=output_tokens,  # [B, T]
                    encoder_out=encoder_out_for_cg,
                )
                scaled_logits = orig_logits + cg_scale * (orig_logits - cg_logits)

            prob = F.log_softmax(scaled_logits, dim=-1) if use_cg else F.log_softmax(orig_logits, dim=-1)
            _scores, _tokens = torch.max(prob, dim=-1)

            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if step_id < steps - 1:
                masked_ratio = 1 - (step_id + 1) / steps
                skeptical_mask = _skeptical_unmasking(
                    output_scores, output_tokens.ne(pad_token), masked_ratio
                )
                output_tokens.masked_fill_(skeptical_mask, unk_token)  # fill in pad for sedd
                output_scores.masked_fill_(skeptical_mask, 0.0)

        # ----------------- Call Vocoder to Synthesize Speech ----------------------- #
        for i in range(len(cur_batch)):
            out_token = output_tokens[i, :tgt_lengths[i]]
            vocoder_inference(args, file_number, "", vocoder_model, out_token)
            file_number += 1



def predictor_loop(model, _tokens, src_feature, src_mask, tgt_mask):
    pred_score = model.encoder.predictor_forward(_tokens,
                                                 src_feature=src_feature,
                                                 src_mask=src_mask,
                                                 tgt_mask=tgt_mask)
    return torch.log(pred_score + 1e-8)




def main(args):
    logger.info(args)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    gen_with_units(args)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hyp_unit", action="store_true", help="use hyp unit from AR system as initial input rather than noise")
    parser.add_argument("--audio_dir", type=str, help="path to audio dir", default=None)
    parser.add_argument("--hyp_unit_file", type=str, help="path to ar hyp unit file", default=None)
    parser.add_argument("--ref_unit_file", type=str, help="path to ar ref unit file", default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit the number of samples to test")
    parser.add_argument("--dummy-config", type=str, help="path to a dummy config file", default=None)
    parser.add_argument("--audio_id", type=str, help="path to audio id file", default=None)
    parser.add_argument("--src_feat_dir", type=str, help="path to source mhubert feats", default=None)
    parser.add_argument("--tgt_feat_dir", type=str, help="path to target vae feats", default=None)
    parser.add_argument("--model_ckpt", type=str, help="path to the diffusion model checkpoint", default=None)
    parser.add_argument("--mhubert_ckpt", type=str, help="path to the mhubert ckpt", default=None)
    # parser.add_argument("--nar_ckpt", type=str, help="path to the nar ckpt", default=None)
    parser.add_argument(
        "--vocoder_ckpt", type=str, help="path to the repr-vocoder ckpt"
    )
    parser.add_argument("--vocoder_cfg", type=str, help="path to the repr-vocoder config", default=None)
    parser.add_argument("--vae_ckpt", type=str, help="path to the vae ckpt", default=None)
    parser.add_argument(
        "--output_dir", type=str, help="path to output dir")
    parser.add_argument("--num_iter", type=int, default=15, help="number of iterations for NAR")
    parser.add_argument("--cg_scale", type=float, default=-1, help="classifier free guidance weight")
    parser.add_argument("--use_sp", action="store_true", help="use sp for nar")
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
