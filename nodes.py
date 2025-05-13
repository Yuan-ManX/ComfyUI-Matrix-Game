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


class VideoGenerator:
    """Main class for video generation using MatrixGame model."""
    
    def __init__(self, dit_path, vae_path, textenc_path, image_path, mouse_icon_path, mouse_scale, mouse_rotation, fps, 
                 output_path, video_length, guidance_scale, inference_steps, shift, num_pre_frames, num_steps, 
                 rel_l1_thresh, resolution_h, resolution_w, bfloat16, max_images, gpu_id):
        """
        Initialize the video generator with configuration parameters.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = FlowMatchDiscreteScheduler(
            shift=self.shift,
            reverse=True,
            solver="euler"
        )
        self.video_length = video_length
        self.guidance_scale = guidance_scale
        
        # Initialize models
        self._init_models()
        
        # Teacache settings
        self._setup_teacache()
    
    def _init_models(self) -> None:
        """Initialize all required models (VAE, text encoder, transformer)."""
        # Initialize VAE
        vae_path = self.vae_path 
        self.vae = get_vae("matrixgame", vae_path, torch.float16)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.enable_tiling()
        
        # Initialize DIT (Transformer)
        dit = MGVideoDiffusionTransformerI2V.from_pretrained(self.dit_path)
        dit.requires_grad_(False)
        dit.eval()
        
        # Initialize text encoder
        textenc_path = self.textenc_path
        weight_dtype = torch.bfloat16 if self.args.bfloat16 else torch.float32
        self.text_enc = get_text_enc('matrixgame', textenc_path, weight_dtype=weight_dtype, i2v_type='refiner')
        
        # Move models to devices
        self.pipeline = MatrixGameVideoPipeline(
            vae=self.vae.vae,
            text_encoder=self.text_enc,
            transformer=dit,
            scheduler=self.scheduler,
        ).to(weight_dtype).to(self.device)
    
    def _setup_teacache(self) -> None:
        """Configure teacache for the transformer."""
        self.pipeline.transformer.__class__.enable_teacache = True
        self.pipeline.transformer.__class__.cnt = 0
        self.pipeline.transformer.__class__.num_steps = self.num_steps  # should be aligned with infer_steps
        self.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
        self.pipeline.transformer.__class__.rel_l1_thresh = self.rel_l1_thresh
        self.pipeline.transformer.__class__.previous_modulated_input = None
        self.pipeline.transformer.__class__.previous_residual = None
        self.pipeline.transformer.__class__.forward = teacache_forward
    
    def _resize_and_crop_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize and crop image to target dimensions while maintaining aspect ratio.
        
        Args:
            image: Input PIL image
            target_size: Target (width, height) tuple
            
        Returns:
            Resized and cropped PIL image
        """
        w, h = image.size
        tw, th = target_size
        
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        
        return image.crop((left, top, right, bottom))
    
    def _load_images(self, root_dir: str) -> List[str]:
        """
        Load image paths from directory with specified extensions.
        
        Args:
            root_dir: Root directory to search for images
            
        Returns:
            List of image file paths
        """
        image_extensions = ('*.png', '*.jpg', '*.jpeg')
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
            
        return image_paths[:self.max_images] if hasattr(self, 'max_images') else image_paths
    
    def _process_condition(self, condition: Dict, image_path: str) -> None:
        """
        Process a single condition and generate video.
        
        Args:
            condition: Condition dictionary containing action and conditions
            image_path: Path to input image
        """
        # Prepare conditions
        keyboard_condition = torch.tensor(condition['keyboard_condition'], dtype=torch.float32).unsqueeze(0)
        mouse_condition = torch.tensor(condition['mouse_condition'], dtype=torch.float32).unsqueeze(0)
        
        # Move to device
        keyboard_condition = keyboard_condition.to(torch.bfloat16 if self.bfloat16 else torch.float16).to(self.device)
        mouse_condition = mouse_condition.to(torch.bfloat16 if self.bfloat16 else torch.float16).to(self.device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        new_width, new_height = self.resolution
        initial_image = self._resize_and_crop_image(image, (new_width, new_height))
        semantic_image = initial_image
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
        initial_image = video_processor.preprocess(initial_image, height=new_height, width=new_width)
        
        if self.args.num_pre_frames > 0:
            past_frames = initial_image.repeat(self.num_pre_frames, 1, 1, 1)
            initial_image = torch.cat([initial_image, past_frames], dim=0)
        
        # Generate video
        with torch.no_grad():
            video = self.pipeline(
                height=new_height,
                width=new_width,
                video_length=self.video_length,
                mouse_condition=mouse_condition,
                keyboard_condition=keyboard_condition,
                initial_image=initial_image,
                num_inference_steps=self.inference_steps if hasattr(self.args, 'inference_steps') else 50,
                guidance_scale=self.guidance_scale,
                embedded_guidance_scale=None,
                data_type="video",
                vae_ver='884-16c-hy',
                enable_tiling=True,
                generator=torch.Generator(device="cuda").manual_seed(42),
                i2v_type='refiner',
                args=self.args,
                semantic_images=semantic_image
            ).videos[0]
        
        # Save video
        img_tensors = rearrange(video.permute(1, 0, 2, 3) * 255, 't c h w -> t h w c').contiguous()
        img_tensors = img_tensors.cpu().numpy().astype(np.uint8)
        
        config = (
            keyboard_condition[0].float().cpu().numpy(),
            mouse_condition[0].float().cpu().numpy()
        )
        
        action_name = condition['action_name']
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{action_name}.mp4"
        output_path = os.path.join(self.output_path, output_filename)
        
        process_video(
            img_tensors,
            output_path,
            config,
            mouse_icon_path=self.mouse_icon_path,
            mouse_scale=self.mouse_scale,
            mouse_rotation=self.mouse_rotation,
            fps=self.fps
        )
            

    
    def generate_videos(self) -> None:
        """Main method to generate videos for all conditions."""
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load conditions
        conditions = Bench_actions_76()
        print(f"Found {len(conditions)} conditions to process")
        
        # Load sample images
        root_dir = self.image_path
        image_paths = self._load_images(root_dir)
        
        if not image_paths:
            print("No images found in the specified directory")
            return
        
        # Process each condition
        for idx, condition in enumerate(conditions):
            for image_path in image_paths:
                print(f"Processing condition {idx+1}/{len(conditions)} with image {os.path.basename(image_path)}")
                self._process_condition(condition, image_path)


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
                "fps": ("INT", {"default": 16}),
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


class GameVideoGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_path": ("DIT",),
                "vae_path": ("VAE",),
                "textenc_path": ("TEXTENCODER",),
                "image_path": ("IMAGE",),
                "mouse_icon_path": ("MOUSEICON",),
                "mouse_scale": ("MOUSESCALE",),
                "mouse_rotation": ("MOUSEROTATION",),
                "fps": ("FPS",),
                "output_path": ("OUTPUT",),
                "video_length": ("INT", {"default": 65}),
                "guidance_scale": ("FLOAT", {"default": 6}),
                "inference_steps": ("INT", {"default": 50}),
                "shift": ("INT", {"default": 15.0}),
                "num_pre_frames": ("INT", {"default": 5}),
                "num_steps": ("INT", {"default": 50}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.075}),
                "resolution_h": ("INT", {"default": 720}),
                "resolution_w": ("INT", {"default": 1280}),
                "bfloat16": ("STRING", {"default": "store_true"}),
                "max_images": ("INT", {"default": 3}),
                "gpu_id": ("STRING", {"default": "0"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "generate_game_videos"
    CATEGORY = "Matrix-Game"

    def generate_game_videos(self, dit_path, vae_path, textenc_path, image_path, mouse_icon_path, mouse_scale, mouse_rotation, fps, 
                             output_path, video_length, guidance_scale, inference_steps, shift, num_pre_frames, num_steps, 
                             rel_l1_thresh, resolution_h, resolution_w, bfloat16, max_images, gpu_id):
        resolution = [resolution_w, resolution_h]
                                 
        generator = VideoGenerator(dit_path, vae_path, textenc_path, image_path, mouse_icon_path, mouse_scale, mouse_rotation, fps, 
                             output_path, video_length, guidance_scale, inference_steps, shift, num_pre_frames, num_steps, 
                             rel_l1_thresh, resolution_h, resolution_w, bfloat16, max_images, gpu_id)
                                 
        generator.generate_videos()
        return ()


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

