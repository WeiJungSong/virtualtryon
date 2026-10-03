"""Tiny randomly initialised CatVTON-shaped models for tests (no downloads)."""
from pathlib import Path


def tiny_unet():
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(
        sample_size=32, in_channels=9, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=4,
    )


def tiny_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                         latent_channels=4, layers_per_block=1, norm_num_groups=4)


def scheduler_dir(tmp: Path) -> Path:
    from diffusers import DDIMScheduler
    d = Path(tmp) / "base"
    DDIMScheduler().save_pretrained(d / "scheduler")
    return d
