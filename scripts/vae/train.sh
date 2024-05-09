# trianing en -> other direction with reduced unit
lang="fr"
data_dir=$cvss_dir/$lang-en/en2${lang}/orig_unit

lr=5e-4
warmup_steps=10000
max_update=200000
max_tokens=15000
update_freq=1
latent_dim=$1

output_dir=$exp_dir/ckpt/speech_vae_decoder_${latent_dim}/en2${lang}/lr${lr}_warmup${warmup_steps}_maxup${max_update}_upfreq${update_freq}
mkdir -p $output_dir

# Note that you need to first prepare the speech feature for the (src)tgt-feat-dir. Note that since VAE is trained on target feature only, so the src-feat-dir command is deprecated but still required.
# The dummy config is used to load some setup for fairseq, our config is available under 



python $project_root/fairseq_cli/train.py \
  $data_dir \
  --tgt-feat-dir $cvss_dir/$lang-en/${lang}/feat/ \
  --src-feat-dir $cvss_dir/$lang-en/en/feat/ \
  --config-yaml config.yaml --dummy-config $YOUR_CONFIG \
  --task speech_decoder --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_vae_decoder_loss \
  --arch speech_vae_decoder --latent_dim $latent_dim \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${output_dir} \
  --keep-best-checkpoints 5 --best-checkpoint-metric "loss" --keep-last-epochs 5 \
  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates $warmup_steps \
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 2.0 \
  --max-update $max_update --max-tokens $max_tokens --max-target-positions 2048 --update-freq $update_freq \
  --seed 42 --num-workers 8 --log-interval 50 \
#   --wandb-project speech-encdec \



