import argparse
import sacrebleu

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="language")
    parser.add_argument("--gen-file", type=str, help="path to manifest file")
    parser.add_argument("--output-dir", type=str, help="path to output")
    parser.add_argument("--manifest", type=str, help="path to manifest file")
    parser.add_argument("--test-file", type=str, help="path to test dir")
    return parser

def main(args):
    file = args.gen_file
    hyps, refs = [], []
    ref_file = f"{args.output_dir}/ref.unit"
    hyp_file = f"{args.output_dir}/hyp.unit"
    ref_handle = open(ref_file, "w")
    hyp_handle = open(hyp_file, "w")
    transcript_file = f"{args.output_dir}/transcript.txt"
    transcript_handle = open(transcript_file, "w")
    # load manifest file
    id2index = {}
    with open(args.test_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.rstrip()
            components = line.split("\t")
            sample_id = components[0]
            id2index[sample_id] = i-1

    # link transcription to the id
    id2trans = {}
    not_found_id =0
    with open(args.manifest, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.rstrip()
            components = line.split("\t")
            path = components[1] if args.lang != "en" else components[0]
            if ".mp3" or ".wav" in path:
                sample_id = path.split(".")[0]
            if not sample_id in id2index:
                not_found_id += 1
                continue
            sample_index = id2index[sample_id]
            transcripts = components[2].rstrip() if args.lang != "en" else components[1].rstrip()
            id2trans[sample_index] = transcripts

    # it is okay, since not all files are used to build parallel data
    print(f"not found id: {not_found_id}")

    prev_hyp, prev_ref = None, None
    prev_sample_id = None
    prev_transcript = None
    error_found = False
    with open(file, "r") as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            try:
                components = line.split("\t")
            except ValueError:
                print(line)
                return

            if components[0].startswith("H-"):
                prev_hyp = components[2]

            elif components[0].startswith("T-"):
                index = components[0].split("\t")[0].split("-")[1]
                index = int(index.strip())
                # retrieve transcriptions using prepared dictionary
                if not index in id2trans:
                    print(f"index {index} not found in manifest file")
                    print("this should not happen!")
                    prev_ref = None
                    continue
                ref_trans = id2trans[index]
                prev_transcript = ref_trans
                prev_ref = components[1]

            elif components[0].startswith("D-"):
                # target and reference are cached in prev..., write them
                if prev_hyp is not None and prev_ref is not None:
                    hyps.append(prev_hyp)
                    refs.append(prev_ref)
                    item_bleu = sacrebleu.corpus_bleu([prev_hyp], [[prev_ref]], lowercase=True).score
                    print(f"score={item_bleu}|{prev_hyp}", file=hyp_handle)
                    print(f"score={item_bleu}|{prev_ref}", file=ref_handle)
                    print(prev_transcript, file=transcript_handle)

                # otherwise something wrong happens when reading the file

        # handle the last hyp-ref pair
        item_bleu = sacrebleu.corpus_bleu([prev_hyp], [[prev_ref]], lowercase=True).score
        print(f"score={item_bleu}|{prev_hyp}", file=hyp_handle)
        print(f"score={item_bleu}|{prev_ref}", file=ref_handle)

    ref_handle.close()
    hyp_handle.close()
    assert len(hyps) == len(refs)
    bleu = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    print(bleu.score)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
