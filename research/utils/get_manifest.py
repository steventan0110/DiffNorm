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
import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument("--split", type=str, default="train", help="split name")
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser

def main(args):
    dest_dir = os.path.dirname(args.dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    with open(args.dest, "w") as train_f:
        print(dir_path, file=train_f)
        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)
            frames = soundfile.info(fname).frames
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=train_f
            )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
