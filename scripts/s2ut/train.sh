# trianing en -> other direction with reduced unit
lang="fr"
start_time=$1
data_dir=$YOUR_SPEECH_UNITS_DIR # this data_dir should contain the speech units for the target language
# e.g. $cvss_dir/$lang-en/en2${lang}/diff_unit_vae_${start_time} for normalized speech units
# or $cvss_dir/$lang-en/en2${lang}/original_unit for the original speech units

lr=5e-4
warmup_steps=10000
max_update=400000
max_tokens=40000
update_freq=1
output_dir=$exp_dir/ckpt/nar_dist_${start_time}/en2${lang}/reduce_unit/lr${lr}_warmup${warmup_steps}_maxup${max_update}_upfreq${update_freq}
mkdir -p $output_dir

python $project_root/fairseq_cli/train.py $data_dir \
  --config-yaml config.yaml \
  --cg_prob 0.0 \
  --task speech_to_speech_fasttranslate --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion nar_speech_to_unit --label-smoothing 0.2 \
  --arch nar_s2ut_conformer --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${output_dir} \
  --keep-best-checkpoints 5 --best-checkpoint-metric "loss" --keep-last-epochs 5 \
  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update ${max_update} --max-tokens ${max_tokens} --max-target-positions 1024 --update-freq $update_freq \
  --seed 42 --num-workers 8 \
  --validate-interval 5 --save-interval 5 \
  --attn-type espnet --pos-enc-type rel_pos \
#   --wandb-project nat-cg-es-ablation \
