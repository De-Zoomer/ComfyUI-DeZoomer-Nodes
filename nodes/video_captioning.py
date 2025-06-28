"""
Video Captioning Node for ComfyUI using Qwen2.5-VL model.
This node generates detailed captions for videos using the Qwen2.5-VL model.
It takes video frames as input and produces a comprehensive description of the video content.
"""

from __future__ import annotations

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import folder_paths
import comfy.model_management as mm
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download


class VideoCaptioningNode:
    """
    A node that generates detailed captions for videos using the Qwen2.5-VL model.
    
    This node processes video frames and generates detailed captions using the Qwen2.5-VL model.
    It supports both 4-bit and 8-bit quantization for different memory requirements.
    The node is designed to work with ComfyUI's IMAGE type input, which should be a sequence of video frames.
    """
    
    def __init__(self):
        """Initialize the VideoCaptioningNode with default configuration."""
        self.model = None
        self.offload_device = mm.unet_offload_device()
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Default configuration
        self.config = {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "max_frames": 768,
            "min_size": 224,
            "max_size": 448,
            "temperature": 0.3,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "num_beams": 1,
            "seed": 123,
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the node.
        
        Returns:
            Dict containing input type definitions for ComfyUI.
        """
        return {
            "required": {
                "images": ("IMAGE",),
                "user_prompt": ("STRING", {
                    "default": """1. **Main Content:**
    * What is the primary focus of the scene?
    * Who are the main characters visible?

2. **Object and Character Details:**
    * Don't refer to characters as 'individual', 'characters' and 'persons', instead always use their gender or refer to them with their gender.
    * Describe the appearance in detail
    * What notable objects are present?

3. **Actions and Movement:**
    * Describe ALL movements, no matter how subtle.
    * Specify the exact type of movement (walking, running, etc.).
    * Note the direction and speed of movements.

4. **Background Elements:**
    * Describe the setting and environment.
    * Note any environmental changes.

5. **Visual Style:**
    * Describe the lighting and color palette.
    * Note any special effects or visual treatments.
    * What is the overall style of the video? (e.g., realistic, animated, artistic, documentary)

6. **Camera Work:**
    * Describe EVERY camera angle change.
    * Note the distance from subjects (close-up, medium, wide shot).
    * Describe any camera movements (pan, tilt, zoom).

7. **Scene Transitions:**
    * How does each shot transition to the next?
    * Note any changes in perspective or viewing angle.

Please be extremely specific and detailed in your description. If you notice any movement or changes, describe them explicitly.""",
                    "multiline": True
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a professional video analyst. Please provide an analysis of this video by covering each of these aspects in your answer. Use only one paragraph. DO NOT separate your answer into topics.",
                    "multiline": True
                }),
                "model_name": (["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-1.5B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct", "Vchitect/ShotVL-7B", "Skywork/SkyCaptioner-V1"], {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.1}),
                "use_flash_attention": ("BOOLEAN", {"default": True}),
                "low_cpu_mem_usage": ("BOOLEAN", {"default": True}),
                "quantization_type": (["4-bit", "8-bit"], {"default": "4-bit"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "videocaptioning"
    CATEGORY = "DeZoomerNodes/text"
    DESCRIPTION = """Generates detailed captions for videos using the Qwen2.5-VL model.

Input Parameters:
- images: Input video frames to process (ComfyUI's IMAGE type)
- user_prompt: Detailed instructions for what aspects to analyze (default provided)
- system_prompt: Instructions for the model's behavior and output style
- model_name: Qwen2.5-VL model to use (7B, 1.5B, or 72B variants)
- temperature: Controls randomness in generation (0.1-1.0)
- use_flash_attention: Enables faster attention implementation
- low_cpu_mem_usage: Optimizes for low CPU memory usage
- quantization_type: Memory optimization (4-bit or 8-bit)
- keep_model_loaded: Whether to keep the model in memory after processing
- seed: Random seed for reproducible generation

The node processes video frames and generates comprehensive descriptions covering:
- Main content and characters
- Object and character details
- Actions and movements
- Background elements
- Visual style
- Camera work
- Scene transitions"""

    def load_model(self, model_name: str, quantization_type: str) -> None:
        """
        Load the Qwen2.5-VL model and processor.
        
        Args:
            model_name: Name of the model to load
            quantization_type: Type of quantization to use ("4-bit" or "8-bit")
        """
        if self.model is None or self.processor is None:
            try:
                # Construct model path in ComfyUI's models directory
                model_checkpoint = os.path.join(
                    folder_paths.models_dir, "LLM", os.path.basename(model_name)
                )

                # Download model if it doesn't exist
                if not os.path.exists(model_checkpoint):
                    print(f"Downloading model {model_name} to {model_checkpoint}...")
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_checkpoint,
                        local_dir_use_symlinks=False,
                    )

                # Load model and processor from local path
                self.processor = AutoProcessor.from_pretrained(
                    model_checkpoint,
                    trust_remote_code=True,
                    min_pixels=128 * 28 * 28,
                    max_pixels=768 * 28 * 28,
                )
                
                # Create quantization config based on selected type
                quantization_config = self.create_quantization_config(quantization_type)
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_checkpoint,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2" if self.config["use_flash_attention"] else None,
                    device_map="auto",
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    use_cache=True,
                    max_memory=None,
                    offload_folder="./offload_folder",
                    low_cpu_mem_usage=self.config["low_cpu_mem_usage"],
                ).eval()
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    def create_quantization_config(self, quantization_type: str) -> BitsAndBytesConfig:
        """
        Create quantization configuration based on the selected type.
        
        Args:
            quantization_type: Type of quantization to use ("4-bit" or "8-bit")
            
        Returns:
            BitsAndBytesConfig object with appropriate settings
        """
        if quantization_type == "4-bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:  # 8-bit
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

    def preprocess_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Preprocess a single frame to fit within size constraints.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame as PIL Image
        """
        # Convert to PIL Image
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        # Resize frame to fit within min/max size constraints
        width, height = frame.size
        if width < height:
            new_width = self.config["min_size"]
            new_height = int(height * (self.config["min_size"] / width))
        else:
            new_height = self.config["min_size"]
            new_width = int(width * (self.config["min_size"] / height))
            
        if new_width > self.config["max_size"] or new_height > self.config["max_size"]:
            if new_width > new_height:
                new_width = self.config["max_size"]
                new_height = int(height * (self.config["max_size"] / width))
            else:
                new_height = self.config["max_size"]
                new_width = int(width * (self.config["max_size"] / height))
                
        frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return frame

    def generate_caption(self, frames: List[np.ndarray], user_prompt: str, system_prompt: str, keep_model_loaded: bool) -> str:
        """
        Generate a caption for the given video frames.
        
        Args:
            frames: List of video frames
            user_prompt: User prompt for caption generation
            system_prompt: System prompt for the model
            keep_model_loaded: Whether to keep the model in memory after processing
            
        Returns:
            Generated caption as string
        """
        # Prepare the messages for the model
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add user message with video frames
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "video",
                    "video": frames,
                    "min_pixels": self.config["min_size"] * 28 * 28,
                    "max_pixels": self.config["max_size"] * 28 * 28,
                    "max_frames": self.config["max_frames"]
                }
            ]
        })

        # Convert messages to model input format
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Debug code to check frame dimensions after processing
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Set random seed if specified
        if self.config["seed"] != -1:
            torch.manual_seed(self.config["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config["seed"])

        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"],
                repetition_penalty=self.config["repetition_penalty"],
                do_sample=self.config["do_sample"],
                num_beams=self.config["num_beams"]
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
                
        caption = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        if not keep_model_loaded:
            print("Offloading model...")
            del self.processor
            del self.model
            self.processor = None
            self.model = None
            mm.soft_empty_cache()
        
        return caption

    def videocaptioning(
        self,
        images: torch.Tensor,
        user_prompt: str,
        system_prompt: str,
        model_name: str,
        temperature: float,
        use_flash_attention: bool,
        low_cpu_mem_usage: bool,
        quantization_type: str,
        keep_model_loaded: bool,
        seed: int
    ) -> Tuple[str]:
        """
        Main function to process video frames and generate captions.
        
        Args:
            images: Input video frames as tensor
            user_prompt: User prompt for caption generation
            system_prompt: System prompt for the model
            model_name: Name of the model to use
            temperature: Temperature for generation
            use_flash_attention: Whether to use flash attention
            low_cpu_mem_usage: Whether to use low CPU memory
            quantization_type: Type of quantization to use
            keep_model_loaded: Whether to keep the model in memory after processing
            seed: Random seed for generation
            
        Returns:
            Tuple containing the generated caption
        """
        # Update config
        self.config.update({
            "model_name": model_name,
            "temperature": temperature,
            "seed": seed,
            "use_flash_attention": use_flash_attention,
            "low_cpu_mem_usage": low_cpu_mem_usage
        })
        
        # Load model
        self.load_model(model_name, quantization_type)
        
        try:
            # Convert tensor to numpy array and process frames
            frames = []
            for i in range(min(len(images), self.config["max_frames"])):
                # Convert tensor to numpy array and ensure correct format
                frame = images[i].cpu().numpy()
                # Convert from [0,1] to [0,255] and ensure correct shape
                frame = (frame * 255).astype(np.uint8)
                # Ensure correct channel order (RGB)
                if frame.shape[2] == 3:  # If already RGB
                    frame = self.preprocess_frame(frame)
                else:  # If BGR or other format
                    frame = np.transpose(frame, (1, 2, 0))
                    frame = self.preprocess_frame(frame)
                frames.append(frame)
            
            if not frames:
                return ("Error: No frames could be processed.",)
                        
            # Generate caption
            caption = self.generate_caption(frames, user_prompt, system_prompt, keep_model_loaded)
            
            return (caption,)
            
        except Exception as e:
            return (f"Error processing video: {str(e)}",) 