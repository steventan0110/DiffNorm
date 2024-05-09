# use to synthesize reduced es/fr speech-units based on diffusion model
lang="fr"
reduce_tsv=$cvss_dir/$lang-en/en2${lang}/reduce_unit
orig_tsv=$cvss_dir/$lang-en/en2${lang}/orig_unit
feature_dir=$cvss_dir/$lang-en/${lang}/feat/

latent_dim=128
diff_ckpt_dir=$exp_dir/ckpt/diff-norm-vae-${latent_dim}-multitask/en2${lang}/lr1e-4_warmup10000_maxup2000000_upfreq1
diff_ckpt=$diff_ckpt_dir/checkpoint_best.pt
echo "Start Generating Units based on trained Diffusion Model at $diff_ckpt "
start_step=$1 # start from 50th diffusion step
echo "Using start step $start_step"

output_dir=$cvss_dir/$lang-en/en2${lang}/diff_unit_vae_${start_step}
mkdir -p $output_dir

# dummy config can be directly copiped without any processing
cp $reduce_tsv/config.yaml $output_dir/config.yaml

PYTHONPATH=$project_root python $project_root/research/TranSpeech/diff_norm_synthesis.py \
  --dummy-config $cvss_dir/$lang-en/en2${lang}/reduce_unit/config.yaml \
  --reduce_tsv_dir $reduce_tsv --orig_tsv_dir $orig_tsv \
  --feature_dir $feature_dir \
  --model_ckpt $diff_ckpt \
  --output_dir ${output_dir} \
  --start_step $start_step


