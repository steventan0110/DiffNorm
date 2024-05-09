# Steven Tan March 5, 2024

import logging
import random
import torch
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq import utils
import torch.nn.functional as F
from fairseq.models.nat.nonautoregressive_transformer import _mean_pooling
from fairseq.data.data_utils import lengths_to_mask
from fairseq.models.speech_to_text.s2t_conformer import S2TConformerEncoder
from fairseq.models.text_to_speech.diff_transformer import (
    DiffusionTransformerModel,
    base_s2st_transformer_encoder_architecture,
)
from fairseq.models.text_to_speech.latent_module import SpeechVAEEncoderDecoder


logger = logging.getLogger(__name__)

@register_model("speech_vae_decoder")
class SpeechVAEDecoder(FairseqEncoderModel):
    """
    The architecture is inspired from previous work on latent diffusion models such as Natural Speech2
    """
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args


    def forward(self, target_feature, target_unit, **model_kwargs):
        tgt_lengths = model_kwargs["tgt_lengths"]
        tgt_mask = lengths_to_mask(tgt_lengths) # B, Ty
        # quantity_loss = self.pred_length(src_tokens, ~src_mask)
        mse_loss, lm_pred, kl_loss = self.encoder(
            target_feature,
            target_unit,
            tgt_mask
        )
        return mse_loss, lm_pred, kl_loss

    def get_normalized_probs(
            self,
            net_output,
            log_probs,
            sample=None,
    ):
        logits = net_output[0]
        # print("logits shape:", logits.shape)
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=False)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=False)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        print(f"Building SpeechVAEDecoder model with latent dim {args.latent_dim}")
        encoder = SpeechVAEEncoderDecoder(dim=768, latent_dim=args.latent_dim)
        return cls(args, encoder)


    @staticmethod
    def add_args(parser):
        DiffusionTransformerModel.add_args(parser)
        # Conformer
        parser.add_argument("--input-feat-per-channel", default=80)
        parser.add_argument("--depthwise-conv-kernel-size", default=31)
        parser.add_argument("--input-channels", default=1)
        parser.add_argument(
            "--attn-type",
            default=None,
            help="If not specified uses fairseq MHA. Other valid option is espnet",
        )
        parser.add_argument(
            "--pos-enc-type",
            default="abs",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )
        parser.add_argument(
            "--classifier_guidance",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--latent_dim",
            type=int,
            default=16,
        )

    def max_positions(self):
        """Maximum input length supported by the model."""
        return self.encoder.max_positions()


@register_model_architecture("speech_vae_decoder", "speech_vae_decoder")
def base_architecture(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.classifier_guidance = getattr(args, "classifier_guidance", 1.0)
    s2s_architecture_base(args)

def s2s_architecture_base(args):
    base_s2st_transformer_encoder_architecture(args) # conformer setup, follow prior work

    # decoder
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
