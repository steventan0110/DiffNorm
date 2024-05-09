#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import sacrebleu


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gen_file", type=str)
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument("--hyp_file", type=str, default=None)
    parser.add_argument("--ref_file", type=str, default=None)

    return parser

def main(args):
    if args.hyp_file is not None and args.ref_file is not None:
        # bleu compute mode
        hyp, ref = [], []
        with open(args.hyp_file, "r") as f:
            for line in f:
                hyp.append(line.strip())
        with open(args.ref_file, "r") as f:
            for line in f:
                ref.append(line.strip())
        assert len(hyp) == len(ref)
        bleu = sacrebleu.corpus_bleu(hyp, [ref])
        print("Corpus bleu score: ", bleu.score)
        return


    dest_dir = os.path.dirname(args.dest)
    hyp, ref = [], []
    with open(args.gen_file, "r") as f:
        for line in f:
            if line.startswith("H-"):
                hyp_line = line.split("\t")[2].strip()
                hyp.append(hyp_line)
            elif line.startswith("T-"):
                ref_line = line.split("\t")[1].strip()
                ref.append(ref_line)

    assert len(hyp) == len(ref)
    # print(hyp[:5], ref[:5])
    bleu = sacrebleu.corpus_bleu(hyp, [ref])
    # print("Corpus bleu score: ", bleu.score)
    ref_out = os.path.join(dest_dir, "reference.txt")
    hyp_out = os.path.join(dest_dir, "hypotheses.txt")
    with open(ref_out, "w") as f:
        for line in ref:
            print(line, file=f)
    with open(hyp_out, "w") as f:
        for line in hyp:
            print(line, file=f)
    return



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
