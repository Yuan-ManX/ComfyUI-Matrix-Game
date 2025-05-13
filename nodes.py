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
    RETURN_NAMES = ("dit_path",)
    FUNCTION = "load_dit"
    CATEGORY = "Matrix-Game"

    def load_dit(self, model_path):
        dit_path = model_path
        return (dit_path,)
    

class LoadVAEModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_path": ("STRING", {"default": "models/matrixgame/vae"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae_path",)
    FUNCTION = "load_vae"
    CATEGORY = "Matrix-Game"

    def load_vae(self, vae_path):
        vae_path = vae_path
        return (vae_path,)


class LoadTextEncoderModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "textencoder_path": ("STRING", {"default": "models/matrixgame"}),
            }
        }

    RETURN_TYPES = ("TEXTENCODER",)
    RETURN_NAMES = ("textencoder",)
    FUNCTION = "load_textencoder"
    CATEGORY = "Matrix-Game"

    def load_textencoder(self, textencoder_path):
        textenc_path = textencoder_path
        return (textenc_path,)


class LoadGameImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "game_image_path": ("STRING", {"default": "initial_image/"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_path",)
    FUNCTION = "load_image"
    CATEGORY = "Matrix-Game"

    def load_image(self, game_image_path):
        image_path = game_image_path
        return (image_path,)


class LoadMouseIcon:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mouse_icon_path": ("STRING", {"default": "models/matrixgame/assets/mouse.png"}),
                "mouse_scale": ("FLOAT", {"default": 0.1}),
                "mouse_rotation": ("FLOAT", {"default": -20}),
                "fps": ("STRING", {"INT": 16}),
            }
        }

    RETURN_TYPES = ("MOUSEICON", "MOUSESCALE", "MOUSEROTATION", "FPS")
    RETURN_NAMES = ("mouse_icon_path", "mouse_scale", "mouse_rotation", "fps")
    FUNCTION = "load_mouse_icon"
    CATEGORY = "Matrix-Game"

    def load_mouse_icon(self, mouse_icon_path, mouse_scale, mouse_rotation, fps):
        mouse_icon_path = mouse_icon_path
        mouse_scale = mouse_scale
        mouse_rotation = mouse_rotation
        fps = fps
        return (mouse_icon_path, mouse_scale, mouse_rotation, mouse_rotation, fps)


class MatrixGameOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_path": ("STRING", {"default": "./test"}),
            }
        }

    RETURN_TYPES = ("OUTPUT",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "load_output_path"
    CATEGORY = "Matrix-Game"

    def load_output_path(self, output_path):
        output_path = output_path
        return (output_path,)

