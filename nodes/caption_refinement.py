"""
Caption Refinement Node for ComfyUI using Qwen2.5 model.
This node refines and enhances captions using the Qwen2.5 model.
It takes a caption as input and produces a refined version with improved coherence and detail.
"""

from __future__ import annotations

import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import folder_paths
from huggingface_hub import snapshot_download
import os


class CaptionRefinementNode:
    """
    A node that refines and enhances captions using the Qwen2.5 model.
    
    This node takes a caption and applies refinement to make it more coherent,
    remove video-specific references, and add clothing details.
    The node is designed to work with ComfyUI's STRING type input, which should be a caption to refine.
    """
    
    def __init__(self):
        """Initialize the CaptionRefinementNode with default configuration."""
        self.model = None
        self.tokenizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Default configuration
        self.config = {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 200,
            "quantization_type": "4-bit",
            "seed": 123,
            "user_prompt": "Could you enhance and refine the following text while maintaining its core meaning:\n\n{text}\n\nPlease limit the response to {max_tokens} tokens."
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
                "caption": ("STRING", {"forceInput": True}),
                "system_prompt": ("STRING", {
                    "default": """You are an AI prompt engineer tasked with helping me modifying a list of automatically generated prompts.

Keep the original text but only do the following modifications:
- you responses should just be the prompt
- Write continuously, don't use multiple paragraphs, make the text form one coherent whole
- do not mention your task or the text itself
- remove references to video such as "the video begins" or "the video features" etc., but keep those sentences meaningful
- mention the clothing details of the characters
- use only declarative sentences""",
                    "multiline": True
                }),
                "model_name": (["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-72B-Instruct"], {"default": "Qwen/Qwen2.5-7B-Instruct"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 200, "min": 50, "max": 1000, "step": 1}),
                "quantization_type": (["4-bit", "8-bit"], {"default": "4-bit"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_caption",)
    FUNCTION = "captionrefinement"
    CATEGORY = "DeZoomerNodes/text"
    DESCRIPTION = """Refines and enhances captions using the Qwen2.5 model.

Input Parameters:
- caption: Input caption to refine (required)
- system_prompt: Instructions for the model's behavior and output style
- model_name: Qwen2.5 model to use (7B, 1.5B, or 72B variants)
- temperature: Controls randomness in generation (0.1-1.0)
- max_tokens: Maximum tokens for refinement output
- quantization_type: Memory optimization (4-bit or 8-bit)
- seed: Random seed for reproducible generation

The node refines captions by:
- Making the text more continuous and coherent
- Removing video-specific references
- Adding clothing details
- Using only declarative sentences"""

    def load_model(self, model_name: str, quantization_type: str) -> None:
        """
        Load the Qwen2.5 model and tokenizer.
        
        Args:
            model_name: Name of the model to load
            quantization_type: Type of quantization to use ("4-bit" or "8-bit")
        """
        if self.model is None or self.tokenizer is None:
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

                # Prepare model loading kwargs
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                }
                
                # Add quantization settings
                if quantization_type == "4-bit":
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    model_kwargs["bnb_4bit_use_double_quant"] = True
                    model_kwargs["bnb_4bit_quant_type"] = "nf4"
                else:  # 8-bit
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["llm_int8_threshold"] = 6.0
                    model_kwargs["llm_int8_has_fp16_weight"] = False
                
                # Load model and tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_checkpoint,
                    **model_kwargs
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.clear_memory()
                raise

    def clear_memory(self) -> None:
        """Clear CUDA cache and force garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

    def captionrefinement(
        self,
        caption: str,
        system_prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        quantization_type: str,
        seed: int
    ) -> Tuple[str]:
        """
        Refine and enhance a caption using the Qwen2.5 model.
        
        Args:
            caption: Input caption to refine
            system_prompt: Instructions for the model's behavior and output style
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for refinement output
            quantization_type: Type of quantization to use
            seed: Random seed for generation
            
        Returns:
            Tuple containing the refined caption
        """
        # Update config
        self.config.update({
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "quantization_type": quantization_type,
            "seed": seed
        })
        
        # Load model
        self.load_model(model_name, quantization_type)
        
        try:
            # Format the user prompt with the actual text and token limit
            formatted_user_prompt = self.config["user_prompt"].format(
                text=caption, 
                max_tokens=self.config["max_tokens"]
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ]

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Prepare inputs
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

            # Set random seed if specified
            if self.config["seed"] != -1:
                torch.manual_seed(self.config["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config["seed"])

            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config["max_tokens"],
                do_sample=True,
                temperature=self.config["temperature"],
                top_p=self.config["top_p"]
            )

            # Extract generated response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return (response.strip(),)
            
        except Exception as e:
            return (f"Error refining caption: {str(e)}",) 