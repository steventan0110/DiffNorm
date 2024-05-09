from typing import Dict, List
import sacrebleu
import pandas as pd
from glob import glob
from pathlib import Path
from utils import retrieve_asr_config, ASRGenerator
from tqdm import tqdm
from argparse import ArgumentParser
import sacrebleu

def merge_tailo_init_final(text):
    """
    Hokkien ASR hypothesis post-processing.
    """
    sps = text.strip().split()
    results = []
    last_syllable = ""
    for sp in sps:
        if sp == "NULLINIT" or sp == "nullinit":
            continue
        last_syllable += sp
        if sp[-1].isnumeric():
            results.append(last_syllable)
            last_syllable = ""
    if last_syllable != "":
        results.append(last_syllable)
    return " ".join(results)


def remove_tone(text):
    """
    Used for tone-less evaluation of Hokkien
    """
    return " ".join([t[:-1] for t in text.split()])


def extract_audio_for_eval(audio_dirpath: str, audio_format: str):
    if audio_format == "n_pred.wav":
        """
        The assumption here is that pred_0.wav corresponds to the reference at line position 0 from the reference manifest
        """
        audio_list = []
        audio_fp_list = glob((Path(audio_dirpath) / "*_pred.wav").as_posix())
        for i in range(len(audio_fp_list)):
            try:
                audio_fp = (Path(audio_dirpath) / f"{i}_pred.wav").as_posix()
                assert (
                    audio_fp in audio_fp_list
                ), f"{Path(audio_fp).name} does not exist in {audio_dirpath}"
            except AssertionError:
                # check the audio with random speaker
                audio_fp = Path(audio_dirpath) / f"{i}_spk*_pred.wav"
                audio_fp = glob(
                    audio_fp.as_posix()
                )  # resolve audio filepath with random speaker
                assert len(audio_fp) == 1
                audio_fp = audio_fp[0]

            audio_list.append(audio_fp)
    else:
        raise NotImplementedError

    return audio_list


def extract_text_for_eval(
    references_filepath: str, reference_format: str, reference_tsv_column: str = None
):
    if reference_format == "txt":
        reference_sentences = open(references_filepath, "r").readlines()
        reference_sentences = [l.strip().lower() for l in reference_sentences]
    elif reference_format == "tsv":
        tsv_df = pd.read_csv(references_filepath, sep="\t", quoting=3)
        reference_sentences = tsv_df[reference_tsv_column].to_list()
        reference_sentences = [l.strip().lower() for l in reference_sentences]
    else:
        raise NotImplementedError

    return reference_sentences


def compose_eval_data(
    audio_dirpath: list,
    audio_format: str,
    references_filepath: str,
    reference_format: str,
    reference_tsv_column: str = None,
    save_manifest_filepath=None,
    limit=None,
):
    """
    Speech matrix decoding pipeline produces audio with the following mask "N_pred.wav" where N is the order of the corresponding input sample
    """

    reference_sentences = extract_text_for_eval(
        references_filepath, reference_format, reference_tsv_column
    )

    nar_dir, nar_cg_dir, nar_diff_dir, nar_diff_cg_dir = audio_dirpath[0], audio_dirpath[1], audio_dirpath[2], audio_dirpath[3]

    reference_sentences = reference_sentences[:limit]
    nar_predicted_audio_fp_list = extract_audio_for_eval(nar_dir, audio_format)[:limit]
    nar_cg_predicted_audio_fp_list = extract_audio_for_eval(nar_cg_dir, audio_format)[:limit]
    nar_diff_predicted_audio_fp_list = extract_audio_for_eval(nar_diff_dir, audio_format)[:limit]
    nar_diff_cg_predicted_audio_fp_list = extract_audio_for_eval(nar_diff_cg_dir, audio_format)[:limit]


    audio_text_pairs = [
        (nar_audio, nar_cg_audio, nar_diff_audio, nar_diff_cg_audio, reference)
        for nar_audio, nar_cg_audio, nar_diff_audio, nar_diff_cg_audio, reference in zip(nar_predicted_audio_fp_list, nar_cg_predicted_audio_fp_list, nar_diff_predicted_audio_fp_list, nar_diff_cg_predicted_audio_fp_list, reference_sentences)
    ]
    # print(audio_text_pairs[0])

    tsv_manifest = pd.DataFrame(audio_text_pairs,
                                columns=["nar_prediction",
                                         "nar_cg_prediction",
                                         "nar_diff_prediction",
                                         "nar_diff_cg_prediction",
                                         "reference"])
    if save_manifest_filepath is not None:
        tsv_manifest.to_csv(save_manifest_filepath, sep="\t", quoting=3)

    return tsv_manifest


def load_eval_data_from_tsv(eval_data_filepath: str):
    """
    We may load the result of `compose_eval_data` directly if needed
    """
    eval_df = pd.from_csv(eval_data_filepath, sep="\t")

    return eval_df


def run_asr_bleu(args):
    all_dirs = args.audio_dirpath.split(",")


    asr_config = retrieve_asr_config(
        args.lang, args.asr_version, json_path=args.config_path
    )

    asr_model = ASRGenerator(asr_config, cache_dirpath=args.cache_dir)

    eval_manifest = compose_eval_data(
        audio_dirpath=all_dirs,
        audio_format=args.audio_format,
        references_filepath=args.reference_path,
        reference_format=args.reference_format,
        reference_tsv_column=args.reference_tsv_column,
        save_manifest_filepath=None,
        limit=args.limit,
    )

    prediction_transcripts = []
    counter = 0
    references = []

    output_handle=open(f"/weka/scratch/jzhan237/diff_s2s/output/examples/{args.lang}.txt", "w")

    for _, eval_pair in tqdm(
        eval_manifest.iterrows(),
        desc="Transcribing predictions",
        total=len(eval_manifest),
    ):
        try:
            nar_text = asr_model.transcribe_audiofile(eval_pair.nar_prediction)
            nar_cg_text= asr_model.transcribe_audiofile(eval_pair.nar_cg_prediction)
            nar_diff_text = asr_model.transcribe_audiofile(eval_pair.nar_diff_prediction)
            nar_diff_cg_text = asr_model.transcribe_audiofile(eval_pair.nar_diff_cg_prediction)

        except:
            print("Error occur")
            continue
        ref_text = eval_pair.reference
        nar_bleu = sacrebleu.sentence_bleu(nar_text, [ref_text]).score
        nar_cg_bleu = sacrebleu.sentence_bleu(nar_cg_text, [ref_text]).score
        nar_diff_bleu = sacrebleu.sentence_bleu(nar_diff_text, [ref_text]).score
        nar_diff_cg_bleu = sacrebleu.sentence_bleu(nar_diff_cg_text, [ref_text]).score
        # print(nar_bleu, nar_cg_bleu, nar_diff_bleu, nar_diff_cg_bleu)
        # if nar_diff_cg_bleu > 15:
        #     print(nar_bleu, nar_cg_bleu, nar_diff_bleu, nar_diff_cg_bleu)

        if nar_diff_cg_bleu > 25 and (
            nar_diff_cg_bleu > nar_diff_bleu and nar_diff_cg_bleu > nar_cg_bleu and nar_diff_cg_bleu > nar_bleu
        ):
            print()
            print("[BLEU]:", nar_diff_cg_bleu, file=output_handle)
            print("[Ref Text]:", ref_text, file=output_handle)
            print("[NAR]:", nar_text, file=output_handle)
            print("[NAR + CG]:", nar_cg_text, file=output_handle)
            print("[NAR + Diff]:", nar_diff_text, file=output_handle)
            print("[NAR + Diff + CG]:", nar_diff_cg_text, file=output_handle)
            print("#############################################", file=output_handle)
            print()
            counter += 1
        if counter == 100:
            break
    output_handle.close()
    return


def main():
    parser = ArgumentParser(
        description="This script computes the ASR-BLEU metric between model's generated audio and the text reference sequences."
    )

    parser.add_argument(
        "--lang",
        help="The target language used to initialize ASR model, see asr_model_cfgs.json for available languages",
        type=str,
    )
    parser.add_argument(
        "--asr_version",
        type=str,
        default="oct22",
        help="For future support we add and extra layer of asr versions. The current most recent version is oct22 meaning October 2022",
    )
    parser.add_argument(
        "--audio_dirpath",
        type=str,
        help="Path to the directory containing the audio predictions from the translation model",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="Path to the file containing reference translations in the form of normalized text (to be compared to ASR predictions",
    )
    parser.add_argument(
        "--reference_format",
        choices=["txt", "tsv"],
        help="Format of reference file. Txt means plain text format where each line represents single reference sequence",
    )
    parser.add_argument(
        "--reference_tsv_column",
        default=None,
        type=str,
        help="If format is tsv, then specify the column name which contains reference sequence",
    )
    parser.add_argument(
        "--audio_format",
        default="n_pred.wav",
        choices=["n_pred.wav"],
        help="Audio format n_pred.wav corresponds to names like 94_pred.wav or 94_spk7_pred.wav where spk7 is the speaker id",
    )
    parser.add_argument(
        "--results_dirpath",
        default=None,
        type=str,
        help="If specified, the resulting BLEU score will be written to this file path as txt file",
    )
    parser.add_argument("--cache_dir", type=str, help="Path to the ASR model cache")
    parser.add_argument("--config_path", type=str, help="Path to the ASR model config")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to evaluate")
    args = parser.parse_args()

    run_asr_bleu(args)

if __name__ == "__main__":
    main()
