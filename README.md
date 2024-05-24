# DiffNorm: Self-Supervised Normalization for Non-autoregressive Speech-to-speech Translation
#### Weiitng Tan, Jingyu Zhang, Lingfeng Shen, Daniel Khashabi, and Philipp Koehn | Johns Hopkins University


PyTorch Implementation of [DiffNorm (arXiv'24)](https://arxiv.org/abs/2405.13274): Self-Supervised Normalization for Non-autoregressive Speech-to-speech Translation.
<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>
<a href="https://arxiv.org/abs/2405.13274" alt="paper"><img src="https://img.shields.io/badge/DiffNorm-Paper-D9AB42" /></a>
<a href="https://www.clsp.jhu.edu/" alt="jhu"><img src="https://img.shields.io/badge/Johns_Hopkins_University-BEC23F" /></a>
<a href="https://twitter.com/weiting_nlp">
  <img src="https://img.shields.io/twitter/follow/weiting_nlp?style=social&logo=twitter"
      alt="follow on Twitter"></a>
</p>

<!-- [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](todo) -->


<!-- We provide our implementation and pretrained models in this repository. -->

## Acknowledgement
This implementation uses parts of the code from the following Github repos:
[Fairseq](https://github.com/facebookresearch/fairseq), [Transpeech](https://github.com/Rongjiehuang/TranSpeech).
For our implementation of diffusion model, we also refer to the repos for various diffusion-based models: [Improved Diffusion](https://github.com/openai/improved-diffusion), [DIT](https://github.com/facebookresearch/DiT), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), [NaturalSpeech2](https://github.com/lucidrains/naturalspeech2-pytorch), [AudioLDM](https://github.com/haoheliu/AudioLDM). We would like to thank authors for open-sourcing their code.



## Environment and Dataset Preparation


### Dependencies
First install the fairseq package by running:
``` bash
pip install --editable ./
```
which requires
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6

In practice we use 
* [PyTorch](http://pytorch.org/) version == 2.2.1
* Python version == 3.10.13
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

### Dataset

We use the publicly available [CVSS](https://github.com/google-research-datasets/cvss) dataset. Please download the audio waveforms as well as their transcriptions in your local directory. Besides the raw waveform, we use the publicly available mHuBERT model to extract speech feature and predict speech units. Please download the mHuBERT model [here](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md), which is pre-trained on VoxPopuli's En, Es, and Fr speech. The K-Means model is also available in the same page, which can be used to predict the speech feature into 1 of 1000 clusters.


Once mHuBERT model and K-means model are downloaded, follow our code under `/scripts/prepare/feature_dump.sh` to perform feature dump and follow `/scripts/prepare/quantize_unit.sh` to quantize features into units. Note that the features need to be saved to efficiently train the VAE model and diffusion model (though theoretically you could encode the feature on-the-fly during training). 

Note that the scripts above provide original speech units. We also need to prepare reduced speech units (which removes consecutive units to make modeling easier). To reduce the units, it can be easily achieved with following function (which we used in our dataset loader):

```python
def _reduce_tgt(self, tokens):
    """
    input: a list of (unreduced) speech unit tokens
    output: reduced speech units, the duration label for each unit, as well as the indices that are not reduced.
    """
    dedup_tokens = []
    duration_label = []
    index_to_keep = []
    accu_duration = 1
    for i, token in enumerate(tokens):
        if i == 0:
            dedup_tokens.append(token)
            index_to_keep.append(i)
        else:
            if token != tokens[i - 1]:
                duration_label.append(accu_duration)
                dedup_tokens.append(token)
                index_to_keep.append(i)
                accu_duration = 1
            else:
                accu_duration += 1
    duration_label.append(accu_duration)
    return dedup_tokens, duration_label, index_to_keep
```

After the preparation, for each speech-to-speech pair, you should have: (1) raw source speech (2) raw target speech (3) target feature from mHuBERT (4) original target speech units (5) reduced target speech units. 

Yes, there is a lot to preprocess as we are buiding a `speech-to-unit + unit-to-speech` system.. Now we are finally ready to train DiffNorm Model for Speech Normalization as well as the downstream `speech-to-unit` model. Luckily, for `unit-to-speech`, we can rely on previously open-sourced model (more details in later sections).

# Diffusion-based Speech Normalization

First, we train a VAE model using speech features and their clustered units prepared from the previous steps. Afterwards, we train the latent diffusion model with freezed VAE model. The training receip are provided below:

## VAE Model Training

Following our script under `/scripts/vae/train.sh`, you will fine-tune a VAE model based on WaveNet and Transformer architecture using the following bash script (our config file for specaugment and transformation can be found in `/scripts/dummy_config.yaml`):
```bash
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
  --seed 42 --num-workers 8 --log-interval 50 
```
For more details of the architecture and training objective, please refer to the paper and relevant code in this repository. Following Fairseq's structure, all our model files are under `/fairseq/models/text_to_speech`, and for VAE training, it is `/fairseq/models/text_to_speech/speech_vae_decoder.py`. All our task files are under `/fairseq/tasks/`, for VAE training it is `/fairseq/models/text_to_speech/speech_decoder_task.py`. All our criterion files are under `/fairseq/criterions/` and for VAE training, it is `/fairseq/criterions/speech_vae_decoder_loss.py`.


## Diffusion Model Training

After training the VAE model, we are ready to train the diffusion model. The script is available at: `/scripts/diffusion/train.sh` and is shown below:

```bash
python $project_root/fairseq_cli/train.py \
  $data_dir \
  --speech_decoder_ckpt $YOUR_VAE_CKPT \
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
```

Here the "--speech_decoder_ckpt" takes in the path for the pre-trained VAE model. The task, model, and criterion files can be found in the same directory as the VAE model. Note that our actual implementation of the model is available in `/fairseq/models/text_to_speech/latent_modules.py`, where we implemented our latent diffusion model and scheduler. We especially thank [@lucidrains](https://github.com/lucidrains) for implementing [NaturalSpeech2](https://github.com/lucidrains/naturalspeech2-pytorch), from which we adapt many modeling functions in our `latent_modules.py`. 


## Normalization with Trained Model
Once the VAE and diffusion model are both trained, we can use DDIM sampler to denoise speech feature and obtain normalized speech units. We use the following script to generate speech units (available at `/scripts/diffusion/unit_gen.sh`):

```bash
PYTHONPATH=$project_root python $project_root/research/TranSpeech/diff_norm_synthesis.py \
  --dummy-config $YOUR_CONFIG \
  --reduce_tsv_dir $reduce_tsv --orig_tsv_dir $orig_tsv \
  --model_ckpt $YOUR_DIFFUSION_CKPT \
  --feature_dir $feature_dir \
  --output_dir ${output_dir} \
  --start_step $start_step
```
This script will call python function (`diff_norm_synthesis`) that (1) inject Gaussian noise based on passed in `start_step` and (2) denoise the feature to reconstruct speech units and dump them to `output_dir`. After this step, we have normalized speech units that can be used to train (**much**) better speech-to-unit (S2UT) models!

## Training S2UT model

The training of speech-to-unit model follows from prior work to use CMLM as the modeling strategy. We introduce classifier-free guidance to CMLM training and it can be achieved with following script (also available at `/scripts/s2ut/train.sh`)


```bash
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
```

Adjust `--cg_prob` for classifier-free guidance. If `--cg_prob` is 0.0 (which it is by default), then encoder representation dropout is disabled and no regularization is used during training. Otherwise, set it to a positve value (e.g. 0.15 throughout our experiments), then classfier-free guidance will be enabled.


## Inference with NAR S2UT model

To evaluate the S2UT model, please follow the scripts in `/scripts/s2ut/eval.sh` which first call `fairseq-generate` to generate speech units with trained S2UT mdodel. Then it calls python script (as shown below) to synthesize waveform from units:

```bash
VOCODER_CKPT=$public_ckpt_dir/hifigan_${lang}/hifigan.ckpt
VOCODER_CFG=$public_ckpt_dir/hifigan_${lang}/config.json
PYTHONPATH=$project_root python $project_root/examples/speech_to_speech/generate_waveform_from_code.py \
  --limit ${limit} \
  --reduce --dur-prediction \
  --in-code-file ${output_dir}/${unit_type}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${waveform_output_dir}/${unit_type}-${limit}
```
Here the VOCODER checkpoint and its configuration can be downloaded from following directories:


Unit config | Unit size | Vocoder language | Dataset | Model
|---|---|---|---|---
mHuBERT, layer 11 | 1000 | En | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
mHuBERT, layer 11 | 1000 | Es | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json)
mHuBERT, layer 11 | 1000 | Fr | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json)

Then the `eval.sh` script will call ASR-BLEU computation as shown below:

```bash
PYTHONPATH=$project_root python $project_root/examples/speech_to_speech/asr_bleu/compute_asr_bleu_custom.py \
  --lang ${lang} \
  --config_path $project_root/examples/speech_to_speech/asr_bleu/asr_model_cfgs.json \
  --cache_dir $hf_cache_dir \
  --audio_dirpath ${waveform_output_dir}/${unit_type}-$limit \
  --reference_path ${output_dir}/transcript.txt  --reference_format "txt" \
  --results_dirpath ${output_dir} \
  --limit ${limit}
```

To compute ASR-BLEU, you also need to download a pre-trained ASR model that is fine-tuned based on HuBERT and CTC objective. The model is available at [here](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_speech/asr_bleu).

Lastly, we also provide another inference script `eval_cg.sh` at the same directory as `eval.sh`. This script enables inference with classifier-free guidance that computes both original and unconditional probability during parallel decoding of CMLM. Note that this script will require a `nar_model_dir` that points to already generated units (after running `eval.sh` ,you will have this directory). This is because, for fair comparison, we also make our improved model using the same length predictor as the base model, but this is not necessary in real application. We will later adapt the classifier-guidance's inference code directly into Fairseq generator as well.


## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@misc{tan2024diffnorm,
      title={DiffNorm: Self-Supervised Normalization for Non-autoregressive Speech-to-speech Translation}, 
      author={Weiting Tan and Jingyu Zhang and Lingfeng Shen and Daniel Khashabi and Philipp Koehn},
      year={2024},
      eprint={2405.13274},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

