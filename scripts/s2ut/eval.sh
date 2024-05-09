# trianing en -> other direction with reduced unit
lang="es"
data_dir=$cvss_dir/$lang-en/en2${lang}/reduce_unit

lr=5e-4
warmup_steps=10000
max_update=400000
max_tokens=40000
update_freq=2
model_dir=$YOUR_S2UT_DIR
#num_iter=15
num_iter=$1
beam_size=1
output_dir=$exp_dir/output/nar-itr${num_iter}-b${beam_size}/en2${lang}/reduce_unit/
mkdir -p $output_dir


#if [ ! -e "${output_dir}/generate-test.txt" ]; then
fairseq-generate $data_dir \
 --gen-subset test --task speech_to_speech_fasttranslate  --path ${model_dir}/checkpoint_best.pt \
 --target-is-code --target-code-size 1000 --vocoder code_hifigan   --results-path ${output_dir} \
 --iter-decode-max-iter $num_iter --iter-decode-eos-penalty 0 --beam 1   --iter-decode-with-beam $beam_size
#fi

echo "Finish decoding units with learned model: $model_dir"
# cmpute bleu score for the generated hyp
#if [ ! -e "${output_dir}/ref.unit" ]; then
  PYTHONPATH=$project_root python $project_root/research/utils/unit_bleu.py \
    --gen-file ${output_dir}/generate-test.txt \
    --test-file ${cvss_dir}/$lang-en/en2${lang}/reduce_unit/test.tsv \
    --manifest ${cvss_dir}/$lang-en/${lang}/test.tsv \
    --output-dir ${output_dir}
#fi
echo "Finish computing BLEU score for the generated unit"


# Synthesize waveform from predicted units
limit=10000
waveform_output_dir=$output_dir/waveform
mkdir -p $waveform_output_dir/ref-$limit
mkdir -p $waveform_output_dir/hyp-$limit

VOCODER_CKPT=$public_ckpt_dir/hifigan_${lang}/hifigan.ckpt
VOCODER_CFG=$public_ckpt_dir/hifigan_${lang}/config.json

# generate ref and hyp waveforms
for unit_type in hyp; do
PYTHONPATH=$project_root python $project_root/examples/speech_to_speech/generate_waveform_from_code.py \
  --limit ${limit} \
  --reduce --dur-prediction \
  --in-code-file ${output_dir}/${unit_type}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${waveform_output_dir}/${unit_type}-${limit}


PYTHONPATH=$project_root python $project_root/examples/speech_to_speech/asr_bleu/compute_asr_bleu_custom.py \
  --lang ${lang} \
  --config_path $project_root/examples/speech_to_speech/asr_bleu/asr_model_cfgs.json \
  --cache_dir $hf_cache_dir \
  --audio_dirpath ${waveform_output_dir}/${unit_type}-$limit \
  --reference_path ${output_dir}/transcript.txt  --reference_format "txt" \
  --results_dirpath ${output_dir} \
  --limit ${limit}
done


