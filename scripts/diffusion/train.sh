# trianing en -> other direction with reduced unit
lang="fr"
data_dir=$cvss_dir/$lang-en/en2${lang}/orig_unit

lr=1e-4
warmup_steps=10000
max_update=2000000
max_tokens=12000
update_freq=1
latent_dim=$1
multitask=True
output_dir=$exp_dir/ckpt/diff-norm-vae-${latent_dim}-multitask/en2${lang}/lr${lr}_warmup${warmup_steps}_maxup${max_update}_upfreq${update_freq}
mkdir -p $output_dir

# this version use mhubert's feature instead of vae as target repr
python $project_root/fairseq_cli/train.py \
  $data_dir \
  --speech_decoder_ckpt $exp_dir/ckpt/speech_vae_decoder_${latent_dim}/en2${lang}/lr5e-4_warmup10000_maxup200000_upfreq1/checkpoint_best.pt \
  --tgt-feat-dir $cvss_dir/$lang-en/${lang}/feat/ \
  --src-feat-dir $cvss_dir/$lang-en/en/feat/ \
  --config-yaml config.yaml --dummy-config $YOUR_CONFIG \
  --task speech_diffusion_discrete --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion ddpm_discrete_loss \
  --arch diff_discrete --latent_dim ${latent_dim} --multitask $multitask \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${output_dir} \
  --keep-best-checkpoints 5 --best-checkpoint-metric "loss" --keep-last-epochs 5 \
  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates $warmup_steps \
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 2.0 \
  --max-update $max_update --max-tokens $max_tokens --max-target-positions 2048 --update-freq $update_freq \
  --validate-interval 5 --save-interval 5 \
  --seed 42 --num-workers 8 --log-interval 50 \
#   --wandb-project diff \
#  --ddp-backend=legacy_ddp \



