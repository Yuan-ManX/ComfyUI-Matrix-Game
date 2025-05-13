import argparse
import os
import glob
from typing import Tuple, Dict, List, Optional
import torch
import numpy as np
from PIL import Image
import imageio
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from safetensors.torch import load_file as safe_load
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from matrixgame.sample.pipeline_matrixgame import MatrixGameVideoPipeline
from matrixgame.model_variants import get_dit
from matrixgame.vae_variants import get_vae
from matrixgame.encoder_variants import get_text_enc
from matrixgame.model_variants.matrixgame_dit_src import MGVideoDiffusionTransformerI2V
from matrixgame.sample.flow_matching_scheduler_matrixgame import FlowMatchDiscreteScheduler
from tools.visualize import process_video
from condtions import Bench_actions_76
from teacache_forward import teacache_forward


class LoadDiTModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_path": ("STRING", {"multiline": False, "default": "models/matrixgame/dit"}),
        }}

    RETURN_TYPES = ("DIT",)
    RETURN_NAMES = ("dit",)
    FUNCTION = "load_dit"
    CATEGORY = "Matrix-Game"

    def load_dit(self, model_path):
        dit = model_path
        return (dit,)
    

class LoadVAEModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_path": ("STRING", {"default": "models/matrixgame/vae"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Matrix-Game"

    def load_vae(self, vae_path):
        vae = vae_path
        return (vae,)

