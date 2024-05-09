
TYPE=hubert
CKPT_PATH=$mhubert_ckpt
LAYER=11


for split in "test" "train" "dev"; do
for lang in "es"; do
    # note that manifest file should already exist
    data_dir=$cvss_dir/$lang-en/$lang/$split
    manifest_file=$exp_dir/cvss/$lang-en/$split.$lang.tsv

    if [ ! -e $manifest_file ]; then
        # note that fairseq requires generating manifest for the speech first
        python $project_root/research/utils/get_manifest.py \
            $data_dir --dest $manifest_file --ext "wav"
    fi

    feature_dir=$cvss_dir/$lang-en/$lang/feat/$split
    mkdir -p $feature_dir
    if [ ! -e ${feature_dir}.manifest.tsv ]; then
        echo "start feature dumpping for $split $lang"
        PYTHONPATH=$project_root python $project_root/examples/textless_nlp/gslm/speech2unit/clustering/dump_feats.py \
        --feature_type $TYPE \
        --checkpoint_path $CKPT_PATH \
        --layer $LAYER \
        --manifest_path $manifest_file \
        --out_features_path $feature_dir
    fi
    echo "finished dumpiing feature of $split $lang with $CKPT_PATH at layer $LAYER"
done
done

