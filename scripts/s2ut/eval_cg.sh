lang="es"
data_dir=$cvss_dir/$lang-en/en2${lang}/reduce_unit

lr=5e-4
warmup_steps=10000
max_update=400000
max_tokens=40000
update_freq=1
MHUBERT_CKPT_PATH=$mhubert_ckpt
KM_MODEL_PATH=$mhubert_quantizer_ckpt

model_type=nar_cg #
echo "Evaluation Model Type: $model_type"
model_dir=$exp_dir/ckpt/${model_type}/en2${lang}/reduce_unit/lr${lr}_warmup${warmup_steps}_maxup${max_update}_upfreq${update_freq}
model_ckpt=${model_dir}/checkpoint_best.pt


num_iter=$1
beam_size=1
limit=10000 # number of samples to evaluate on
output_dir=$exp_dir/output/${model_type}-itr${num_iter}-b${beam_size}/en2${lang}/reduce_unit/
mkdir -p $output_dir

# prepare data, make it consistent with baseline model's output and share the same length
nar_model_dir=$exp_dir/output/nar-itr15-b1/en2${lang}/reduce_unit/
TRANSCRIPT=${nar_model_dir}/transcript.txt
GEN_TXT=${nar_model_dir}/generate-test.txt
if [ ! -e ${output_dir}/transcript.txt ]; then
PYTHONPATH=$project_root python $project_root/research/utils/repr_bleu.py \
  --gen-file ${GEN_TXT} \
  --test-file ${cvss_dir}/$lang-en/en2${lang}/reduce_unit/test.tsv \
  --manifest ${cvss_dir}/$lang-en/${lang}/test.tsv \
  --output-dir ${output_dir}
fi


echo "Finish preparing data, start evaluation"
vocoder_ckpt=$public_ckpt_dir/hifigan_${lang}/hifigan.ckpt
vocoder_cfg=$public_ckpt_dir/hifigan_${lang}/config.json


#for cg_scale in 0.0 0.5 1.0 2.0 3.0; do
for cg_scale in 0.5; do
waveform_dir=${output_dir}/waveform-${limit}-${cg_scale}
mkdir -p $waveform_dir

gen_wave=true
compute_asr=true
if $gen_wave; then
rm -rf $waveform_dir
PYTHONPATH=$project_root python $project_root/research/TranSpeech/nat_gen.py \
  --limit ${limit} \
  --dummy-config $cvss_dir/$lang-en/en2${lang}/reduce_unit/config.yaml \
  --audio_id ${output_dir}/audio_id.txt \
  --audio_dir $cvss_dir/$lang-en/en/test \
  --src_feat_dir $cvss_dir/$lang-en/en/feat/test \
  --tgt_feat_dir $cvss_dir/$lang-en/${lang}/feat/test \
  --model_ckpt $model_ckpt \
  --mhubert_ckpt $KM_MODEL_PATH \
  --vocoder_ckpt $vocoder_ckpt --vocoder_cfg $vocoder_cfg \
  --output_dir ${waveform_dir} \
  --cg_scale $cg_scale --num_iter $num_iter \
  --use_hyp_unit \
  --hyp_unit_file $output_dir/hyp_unit.txt --ref_unit_file $output_dir/ref_unit.txt
fi

if $compute_asr; then
echo "Start Evaluating the generated sound waves"
# score the generated waveform
PYTHONPATH=$project_root python $project_root/examples/speech_to_speech/asr_bleu/compute_asr_bleu_custom.py \
  --lang ${lang} \
  --config_path $project_root/examples/speech_to_speech/asr_bleu/asr_model_cfgs.json \
  --cache_dir $hf_cache_dir \
  --audio_dirpath ${waveform_dir}\
  --reference_path ${output_dir}/transcript.txt  --reference_format "txt" \
  --results_dirpath ${output_dir} \
  --limit ${limit}

fi

done

